import argparse
from pathlib import Path
import os
import tarfile
import shutil
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, Tuple

import h5py
import numpy as np
import pydicom
from tqdm import tqdm
import hdf5plugin


# -------------------------------------------------
# Logging
# -------------------------------------------------

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def accession_from_path(p: Path) -> str:
    name = p.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return p.stem


def safe_name(name) -> str:
    if name is None:
        return "unknown"
    cleaned = "".join(c if c.isalnum() else "_" for c in str(name))
    return cleaned if cleaned else "unknown"


def is_safe_member_path(base_dir: Path, member_name: str) -> bool:
    target_path = (base_dir / member_name).resolve()
    try:
        target_path.relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def safe_extract(tf: tarfile.TarFile, path: Path) -> None:
    for member in tf.getmembers():
        if not is_safe_member_path(path, member.name):
            raise RuntimeError(f"Unsafe tar member path detected: {member.name}")
    tf.extractall(path)


def get_chunk_shape(
    volume_shape: Tuple[int, int, int],
    chunk_size: int
) -> Tuple[int, int, int]:
    z, y, x = volume_shape
    z_chunk = max(1, min(chunk_size, z))
    return (z_chunk, y, x)


# -------------------------------------------------
# Convert one accession
# -------------------------------------------------

def convert_accession(args, compression, chunk_size):
    src_path, out_root = args
    src_path = Path(src_path)
    out_root = Path(out_root)

    accession = accession_from_path(src_path)
    tmp_base = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_dir = tmp_base / f"untar_{accession}_{os.getpid()}"

    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        tmp_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(src_path, "r:*") as tf:
            safe_extract(tf, tmp_dir)

        dicoms = []
        mrn = "unknown"

        for root, _, files in os.walk(tmp_dir):
            for f in files:
                path = Path(root) / f

                try:
                    ds = pydicom.dcmread(path, force=True)

                    if not hasattr(ds, "PixelData"):
                        continue

                    arr = ds.pixel_array

                    if arr is None or arr.ndim != 2:
                        continue

                    inst = int(getattr(ds, "InstanceNumber", 0) or 0)

                    if mrn == "unknown":
                        mrn = safe_name(getattr(ds, "PatientID", "unknown"))

                    dicoms.append((inst, arr))

                except Exception:
                    continue

        if not dicoms:
            logging.warning(f"{accession}: no readable 2D DICOM slices found")
            return None

        dicoms.sort(key=lambda x: x[0])

        reference_shape = dicoms[0][1].shape
        filtered_dicoms = [d for d in dicoms if d[1].shape == reference_shape]

        if not filtered_dicoms:
            logging.warning(f"{accession}: no consistent slice shapes after filtering")
            return None

        dropped = len(dicoms) - len(filtered_dicoms)
        if dropped > 0:
            logging.warning(
                f"{accession}: dropped {dropped} slices with mismatched shape; kept {len(filtered_dicoms)}"
            )

        volume = np.stack([d[1] for d in filtered_dicoms], axis=0)
        chunks = get_chunk_shape(volume.shape, chunk_size)

        mrn_dir = out_root / mrn
        mrn_dir.mkdir(parents=True, exist_ok=True)

        output_h5 = mrn_dir / f"{accession}.h5"

        # Skip if already converted
        if output_h5.exists():
            logging.info(f"{accession}: already exists at {output_h5}, skipping")
            return str(output_h5)

        with h5py.File(output_h5, "w") as h5f:
            if compression == "blosc":
                h5f.create_dataset(
                    "volume",
                    data=volume,
                    chunks=chunks,
                    **hdf5plugin.Blosc(
                        cname="zstd",
                        clevel=5,
                        shuffle=hdf5plugin.Blosc.SHUFFLE,
                    ),
                )
            elif compression == "gzip":
                h5f.create_dataset(
                    "volume",
                    data=volume,
                    chunks=chunks,
                    compression="gzip",
                )
            else:
                h5f.create_dataset(
                    "volume",
                    data=volume,
                    chunks=chunks,
                )

            h5f.attrs["accession"] = accession
            h5f.attrs["mrn"] = mrn
            h5f.attrs["num_slices"] = int(volume.shape[0])
            h5f.attrs["height"] = int(volume.shape[1])
            h5f.attrs["width"] = int(volume.shape[2])

        logging.info(f"Created {output_h5}")
        return str(output_h5)

    except Exception as e:
        logging.error(f"{accession} failed: {e}")
        return None

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# -------------------------------------------------
# Batch conversion
# -------------------------------------------------

def convert_all(
    dicoms_dir: Path,
    out_root: Path,
    workers: int,
    compression: str,
    chunk_size: int,
    limit: Optional[int],
    index: int,
    num_jobs: int,
):
    tar_files = sorted(dicoms_dir.glob("*.tar.gz"))

    if limit is not None:
        tar_files = tar_files[:limit]

    if not tar_files:
        print(f"No .tar.gz files found in {dicoms_dir}")
        return

    if index < 0 or index >= num_jobs:
        raise ValueError(f"--index must be in [0, {num_jobs - 1}], got {index}")

    # -----------------------------
    # SLURM PARTITIONING
    # -----------------------------
    total = len(tar_files)
    files_per_job = total // num_jobs

    start = index * files_per_job
    end = (index + 1) * files_per_job if index != num_jobs - 1 else total

    tar_subset = tar_files[start:end]

    print(f"[JOB {index}] Processing {len(tar_subset)} files ({start}:{end})")
    logging.info(f"[JOB {index}] Processing {len(tar_subset)} files ({start}:{end})")

    if not tar_subset:
        print(f"[JOB {index}] No files assigned")
        return

    # -----------------------------
    # SKIP ALREADY DONE FILES
    # -----------------------------
    args_list = []
    skipped = 0

    for p in tar_subset:
        accession = accession_from_path(p)
        if list(out_root.rglob(f"{accession}.h5")):
            skipped += 1
            continue
        args_list.append((p, out_root))

    print(f"[JOB {index}] Skipping {skipped} already converted files")
    logging.info(f"[JOB {index}] Skipping {skipped} already converted files")

    if not args_list:
        print(f"[JOB {index}] Nothing to process (all skipped)")
        return

    # -----------------------------
    # MULTIPROCESSING
    # -----------------------------
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fn = partial(
            convert_accession,
            compression=compression,
            chunk_size=chunk_size,
        )

        results = list(
            tqdm(
                ex.map(fn, args_list),
                total=len(args_list),
                desc=f"Job {index}",
            )
        )

    written = sum(1 for r in results if r is not None)
    print(f"[JOB {index}] Wrote {written} H5 files to {out_root}")
    logging.info(f"[JOB {index}] Wrote {written} H5 files to {out_root}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit_accessions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--compression",
        type=str,
        default="blosc",
        choices=["blosc", "gzip", "none"],
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/gpfs/data/oermannlab/private_data/thoracic/CT/dicom",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1000,
    )

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    dicoms_dir = Path(args.input_dir)

    if args.out_dir is None:
        output_root = dicoms_dir.parent / "h5"
    else:
        output_root = Path(args.out_dir)

    output_root.mkdir(parents=True, exist_ok=True)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", args.max_workers))

    convert_all(
        dicoms_dir=dicoms_dir,
        out_root=output_root,
        workers=workers,
        compression=args.compression,
        chunk_size=args.chunk_size,
        limit=args.limit_accessions,
        index=args.index,
        num_jobs=args.num_jobs,
    )