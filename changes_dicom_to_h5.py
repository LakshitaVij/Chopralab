import argparse
from pathlib import Path
import h5py
import numpy as np
import pydicom
from pydicom import config as pydicom_config
import os
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from functools import partial


pydicom_config.use_gdcm = False

SKIP_LIST: list[str] = []

# Create logs folder
log_folder = Path("logs")
log_folder.mkdir(exist_ok=True)

# Generate log file name with date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_folder / f"conversion_tar_to_h5_{current_time}.log"

# Set up logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def safe_str_conversion(value) -> str:
    """Safely convert any DICOM value to string, handling special cases."""
    if value is None:
        return ""
    try:
        if isinstance(value, (pydicom.sequence.Sequence, list)):
            return f"Sequence with {len(value)} items"
        if isinstance(value, pydicom.multival.MultiValue):
            return ",".join(str(x) for x in value)
        return str(value)
    except Exception as e:
        logging.warning(f"Error converting value to string: {e}")
        return ""

def h5_safe_key(name: str) -> str:
    """Make a key safe for HDF5 dataset names."""
    k = name.replace("/", "_").strip()
    return k if k else "UNKNOWN"

def extract_all_metadata(ds: pydicom.dataset.Dataset) -> dict:
    """
    Iterate ALL top-level DICOM elements and turn them into a flat dict.
    Skips PixelData (stored separately). Uses keyword when available else hex tag.
    """
    meta = {}
    for elem in ds:
        if elem.keyword == "PixelData":
            continue
        key = elem.keyword if elem.keyword else f"{int(elem.tag):08X}"
        key = h5_safe_key(key)
        value = safe_str_conversion(elem.value)
        if key in meta:
            suffix = 2
            while f"{key}_{suffix}" in meta:
                suffix += 1
            key = f"{key}_{suffix}"
        meta[key] = value
    return meta


def convert_dicom_to_h5(
    dicom_path: Path,
    output_folder: Path,
    debug_metadata: bool = False,
) -> None:
    """Converts all DICOM files under dicom_path (recursively) to H5."""
    created_files = False
    try:
        for root, _, files in os.walk(dicom_path):
            for fname in files:
                file = Path(root) / fname
                if not file.is_file():
                    continue  # just in case

                # Process DICOM file from disk
                ds = pydicom.dcmread(file, stop_before_pixels=False, force=True)

                # Try decoding pixel data
                image_data = None
                try:
                    if hasattr(ds, "PixelData"):
                        image_data = ds.pixel_array
                except Exception as e:
                    logging.warning(f"Failed to decode pixel_array for {file}: {e}")

                metadata = extract_all_metadata(ds)

                # Always write an H5: either decoded "image" or raw "pixeldata_raw"
                h5_file_path = output_folder / f"{file.stem}.h5"
                with h5py.File(h5_file_path, "w") as h5f:
                    if image_data is not None:
                        h5f.create_dataset("image", data=image_data, compression="gzip")
                    else:
                        if hasattr(ds, "PixelData"):
                            h5f.create_dataset(
                                "pixeldata_raw",
                                data=np.frombuffer(ds.PixelData, dtype=np.uint8),
                            )
                    metadata_group = h5f.create_group("metadata")
                    for key, value in metadata.items():
                        metadata_group.create_dataset(key, data=str(value))

                logging.info(f"Created H5 file: {h5_file_path}")
                created_files = True

                if debug_metadata:
                    keys_preview = list(metadata.keys())[:40]
                    logging.info(
                        f"[DEBUG] {h5_file_path.name} metadata keys (first 40): {keys_preview}"
                    )

    except Exception as e:
        logging.error(f"Error processing {dicom_path}: {e}")

    # Cleanup empty folder
    if not created_files and output_folder.exists():
        try:
            output_folder.rmdir()
            logging.warning(f"No H5 files created, removed empty folder: {output_folder}")
        except OSError:
            logging.warning(f"No H5 files created; could not remove folder: {output_folder}")


def process_dicom_folder(args: tuple, accession_to_mrn: dict, debug_metadata: bool = False) -> None:
    """Process a single accession folder using a precomputed accession->MRN map."""
    dicom_path, output_h5_dir = args

    # Use the directory name as the accession key
    accession = dicom_path.name
    key = accession.strip()

    pat_mrn_id = str(accession_to_mrn.get(key, "unknown"))
    if pat_mrn_id == "unknown":
        logging.warning(f"No MRN mapping for accession {key}")

    output_folder = output_h5_dir / pat_mrn_id / accession

    # Skip if the output folder already exists and is not empty (resume)
    if output_folder.exists() and any(output_folder.iterdir()):
        logging.info(
            f"Skipping {dicom_path}, output folder {output_folder} already exists and is not empty."
        )
        print(
            f"Skipping {dicom_path}, output folder {output_folder} already exists and is not empty."
        )
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    try:
        convert_dicom_to_h5(
            dicom_path=dicom_path,
            output_folder=output_folder,
            debug_metadata=debug_metadata,
        )
    except Exception as e:
        logging.error(f"Error processing accession {accession}: {e}")
        return  # swallow the error so one bad accession doesn't kill the pool


def convert_all_dicoms_to_h5(
    dicoms_dir: Path,
    output_h5_dir: Path,
    max_workers: int,
    accession_to_mrn: dict,
    processed_accs: list,
    debug_metadata: bool = False,
    limit: int | None = None
) -> None:
    """Converts all accession directories of DICOMs to H5 using parallel processing."""
    # Only process directories (ignore tar.gz files if present)
    dicom_folders = [p for p in dicoms_dir.iterdir() if p.is_dir()]
    args = [
        (dicom, output_h5_dir)
        for dicom in dicom_folders
        if str(dicom) not in SKIP_LIST and dicom.name not in processed_accs
    ]
    if limit is not None:
        args = args[:limit]
    print(f"Number of dicoms to process {len(args)}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        map_fn = partial(
            process_dicom_folder,
            accession_to_mrn=accession_to_mrn,
            debug_metadata=debug_metadata
        )
        mapped_results = executor.map(map_fn, args)

        for i, _ in enumerate(
            tqdm(
                mapped_results,
                total=len(args),
                desc="Converting dicom files to H5",
                unit="file",
            ),
            1,
        ):
            # Periodically count folders and print.
            if i % 100 == 0:
                folder_count = sum(1 for _p in output_h5_dir.iterdir() if _p.is_dir())
                print(
                    f"Processed {i} files. Current folder count: {folder_count} / {len(args)}"
                )

def list_processed_accs(output_images_root: str) -> list:
    """
    Return a flat list of accession IDs that already have output under:
        <output_images_root>/<MRN>/<ACC>/
    """
    processed = []
    root = Path(output_images_root)
    if not root.exists():
        return processed
    for mrn_dir in root.iterdir():       # MRN level
        if not mrn_dir.is_dir():
            continue
        for acc_dir in mrn_dir.iterdir():  # ACC level
            if acc_dir.is_dir():
                processed.append(acc_dir.name)
    return processed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert dicom files to H5 files.")
    parser.add_argument(
        "--report-csv",
        type=str,
        help="Report CSV path (required if --directory is a raw path)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of workers for parallel processing (default: 8)",
    )
    parser.add_argument(
        "--debug-metadata",
        action="store_true",
        help="Log a preview of metadata keys for each written H5 (helps test extract_all_metadata).",
    )
    parser.add_argument(
        "--directory", 
        type=str,
        required=True,
        help="Path to directory, choose between cxr and ct, or pass a full directory path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: only process this many accession folders (for testing)."
    )
    return parser.parse_args()

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    args = parse_arguments()

    max_workers = args.max_workers
    debug_metadata = args.debug_metadata
    choice = args.directory.lower()

    # Pick directory + report based on choice
    if choice == "cxr":
        dicoms_dir = Path(os.environ.get(
            "CXR_DIR",
            "/gpfs/data/oermannlab/private_data/thoracic/cxr/dicom"
        ))
        report_path = "/gpfs/data/oermannlab/private_data/thoracic/dataset_cxr_nov2024.csv"

    elif choice == "ct":
        dicoms_dir = Path(os.environ.get(
            "CT_DIR",
            "/gpfs/data/oermannlab/private_data/thoracic/ct/dicom"
        ))
        report_path = "/gpfs/data/oermannlab/private_data/thoracic/dataset_tomo_nov2024.csv"

    else:
        # custom path case
        dicoms_dir = Path(args.directory).expanduser().resolve()
        report_path = args.report_csv or os.environ.get("REPORT_PATH")
        if not report_path:
            raise ValueError("Custom --directory requires --report-csv or REPORT_PATH.")

    # normalize & validate
    dicoms_dir = dicoms_dir.expanduser().resolve()
    if not dicoms_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicoms_dir}")
    if not Path(report_path).expanduser().exists():
        raise FileNotFoundError(f"Report CSV not found: {report_path}")

    # -------- Build accession -> MRN map ONCE --------
    report_df = pd.read_csv(report_path, dtype=str).fillna("")

    def _norm(x): 
        return str(x).strip()

    cxr = { _norm(a): _norm(m)
            for a, m in zip(report_df.get("cxr_accession_num", []),
                            report_df.get("pat_mrn_id", [])) if a }
    ct  = { _norm(a): _norm(m)
            for a, m in zip(report_df.get("tomo_accession_num", []),
                            report_df.get("pat_mrn_id", [])) if a }

    accession_to_mrn = {**cxr, **ct}
    logging.info(f"Loaded MRN map: {len(accession_to_mrn)} entries")

    # Output under the modality folder (next to 'dicom')
    output_h5_dir = dicoms_dir.parent / "h5conv"
    output_h5_dir.mkdir(parents=True, exist_ok=True)

    processed_accs = list_processed_accs(str(output_h5_dir))

    convert_all_dicoms_to_h5(
        dicoms_dir=dicoms_dir,
        output_h5_dir=output_h5_dir,
        max_workers=max_workers,
        accession_to_mrn=accession_to_mrn,
        processed_accs=processed_accs,
        debug_metadata=debug_metadata,
        limit=args.limit
    )
