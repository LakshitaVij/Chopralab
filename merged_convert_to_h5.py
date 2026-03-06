# convert_to_h5.py

import argparse
import os
import tarfile
from io import BytesIO
from typing import List, Optional

import numpy as np
import h5py
import pydicom

try:
    import hdf5plugin
    HAS_BLOSC = True
except ImportError:
    HAS_BLOSC = False

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def is_dicom_member(member_name: str) -> bool:
    name = member_name.lower()
    return (
        name.endswith((".dcm", ".dicom", ".ima"))
        or "/dicom" in name
        or not os.path.splitext(name)[1]
    )


def safe_accession_id(tar_path: str) -> str:
    base = os.path.basename(tar_path)
    for suf in [".tar.gz", ".tgz", ".tar"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)


# -----------------------------------------------------------
# DICOM Reading
# -----------------------------------------------------------

def read_dicoms_from_tar(
    tar_path: str,
    max_files: Optional[int] = None
) -> List[pydicom.dataset.FileDataset]:

    dicoms = []
    mode = "r:gz" if tar_path.endswith((".tar.gz", ".tgz")) else "r:"

    with tarfile.open(tar_path, mode) as tf:

        members = [m for m in tf.getmembers() if m.isfile()]
        candidates = [m for m in members if is_dicom_member(m.name)]

        if not candidates:
            candidates = members

        if max_files is not None:
            candidates = candidates[:max_files]

        for m in candidates:
            f = tf.extractfile(m)
            if f is None:
                continue

            raw = f.read()

            try:
                ds = pydicom.dcmread(BytesIO(raw), force=True)
                if getattr(ds, "PixelData", None) is None:
                    continue
                dicoms.append(ds)
            except Exception:
                continue

    return dicoms


def sort_dicoms(ds_list):

    def get_key(ds):
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            return (0, int(inst))

        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) == 3:
            return (1, float(ipp[2]))

        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            return (2, float(sl))

        return (3, 0)

    return sorted(ds_list, key=get_key)


def pixel_array(ds):

    arr = ds.pixel_array

    if arr.dtype not in (np.int16, np.uint16, np.uint8, np.float32):
        arr = arr.astype(np.int16, copy=False)

    return arr




# -----------------------------------------------------------
# H5 Writing
# -----------------------------------------------------------

def write_h5(
    out_path: str,
    accession_id: str,
    ds_list,
    compression_mode: str,
    chunk_size:int
):

    arrays = [pixel_array(ds) for ds in ds_list]
    shapes = [a.shape for a in arrays]

    if len(set(shapes)) != 1:
        from collections import Counter
        mode_shape = Counter(shapes).most_common(1)[0][0]
        arrays = [a for a in arrays if a.shape == mode_shape]
        if not arrays:
            return

    stack = np.stack(arrays, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as hf:

        kwargs = {}

        if compression_mode == "gzip":
            kwargs["compression"] = "gzip"
            kwargs["compression_opts"] = 4
            kwargs["shuffle"] = True

        elif compression_mode == "blosc":
            if not HAS_BLOSC:
                raise RuntimeError("Install hdf5plugin.")
            kwargs.update(
                hdf5plugin.Blosc(
                    cname="zstd",
                    clevel=5,
                    shuffle=hdf5plugin.Blosc.SHUFFLE
                )
            )
        chunk_depth = min(chunk_size, stack.shape[0])

        hf.create_dataset(
            "images",
            data=stack,
            chunks=(chunk_depth, stack.shape[1], stack.shape[2]),
            **kwargs
        )

        hf.attrs["accession_id"] = accession_id
        hf.attrs["num_images"] = int(stack.shape[0])
        hf.attrs["shape_hw"] = str(tuple(stack.shape[-2:]))
        hf.attrs["dtype"] = str(stack.dtype)


# -----------------------------------------------------------
# Conversion
# -----------------------------------------------------------

def convert_one_tar(
    tar_path,
    out_dir,
    chunk_size,
    max_dicoms,
    compression_mode
):

    accession_id = safe_accession_id(tar_path)

    ds_list = read_dicoms_from_tar(tar_path, max_files=max_dicoms)

    if not ds_list:
        return []

    ds_list = sort_dicoms(ds_list)


    out_path = os.path.join(
    out_dir,
    f"{accession_id}.h5"
    )

    write_h5(
        out_path,
        accession_id,
        ds_list,
        compression_mode,
        chunk_size
    )

    return [out_path]

 


def list_tar_files(input_dir):

    out = []

    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith((".tar", ".tar.gz", ".tgz")):
                out.append(os.path.join(root, fn))

    return sorted(out)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_accessions", type=int, default=None)
    ap.add_argument("--max_dicoms_per_accession", type=int, default=None)
    ap.add_argument(
    "--chunk_size",
    type=int,
    default=32,
    help="Number of slices per HDF5 chunk."
)

    ap.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "gzip", "blosc"]
    )

    args = ap.parse_args()

    tar_paths = list_tar_files(args.input_dir)

    if args.limit_accessions is not None:
        tar_paths = tar_paths[:args.limit_accessions]

    total_written = 0

    for i, tar_path in enumerate(tar_paths, 1):

        written = convert_one_tar(
            tar_path=tar_path,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
            max_dicoms=args.max_dicoms_per_accession,
            compression_mode=args.compression
        )

        total_written += len(written)

        print(
            f"[{i}/{len(tar_paths)}] "
            f"{os.path.basename(tar_path)} -> {len(written)} h5"
        )

    print(f"Done. Wrote {total_written} H5 files to {args.out_dir}")


if __name__ == "__main__":
    main()