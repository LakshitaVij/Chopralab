import argparse
from pathlib import Path
import os
import tarfile
import shutil
import logging
from datetime import datetime
from typing import Optional, Tuple
import fcntl

import h5py
import numpy as np
import pydicom
pydicom.config.convert_wrong_length_to_UN = True
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
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if p.name.endswith(suffix):
            return p.name[:-len(suffix)]
    return p.stem


def safe_name(name):
    return "".join(c if c.isalnum() else "_" for c in str(name or "unknown"))


def safe_metadata_value(value):
    if value is None:
        return ""
    try:
        if isinstance(value, pydicom.sequence.Sequence):
            return f"Sequence[{len(value)}]"
        if isinstance(value, list):
            return f"List[{len(value)}]"
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return ",".join(str(x) for x in value)
        return str(value)
    except:
        return "<unreadable>"


def safe_key(elem):
    return safe_name(elem.keyword if elem.keyword else str(elem.tag))


def safe_extract(tf, path):
    tf.extractall(path)


def get_chunk_shape(shape, chunk_size):
    z, y, x = shape
    return (max(1, min(chunk_size, z)), y, x)


def get_run_state_paths(out_root):
    run_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    out_root = Path(out_root)
    return (
        out_root / f".run_{run_id}.lock",
        out_root / f".run_{run_id}.count",
    )


def read_count(count_file: Path) -> int:
    if not count_file.exists():
        return 0
    try:
        text = count_file.read_text().strip()
        return int(text) if text else 0
    except:
        return 0


def write_count(count_file: Path, value: int):
    count_file.write_text(str(value))


# -------------------------------------------------
# Convert one accession
# -------------------------------------------------

def convert_accession(args, compression, chunk_size):
    src_path, out_root = args

    accession = accession_from_path(src_path)
    tmp_dir = Path("/tmp") / f"tmp_{accession}_{os.getpid()}"

    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)

        with tarfile.open(src_path, "r:*") as tf:
            safe_extract(tf, tmp_dir)

        dicoms = []
        mrn = "unknown"

        for root, _, files in os.walk(tmp_dir):
            for f in files:
                try:
                    path = Path(root) / f
                    ds = pydicom.dcmread(path, force=True)

                    if not hasattr(ds, "PixelData"):
                        continue

                    arr = ds.pixel_array
                    if arr is None or arr.ndim != 2:
                        continue

                    inst = int(getattr(ds, "InstanceNumber", 0) or 0)

                    if mrn == "unknown":
                        mrn = safe_name(getattr(ds, "PatientID", "unknown"))

                    dicoms.append((inst, arr, ds))

                except:
                    continue

        if not dicoms:
            logging.warning(f"{accession}: no readable DICOM slices")
            return None

        dicoms.sort(key=lambda x: x[0])
        ref_shape = dicoms[0][1].shape
        dicoms = [d for d in dicoms if d[1].shape == ref_shape]

        if not dicoms:
            logging.warning(f"{accession}: no consistent slices")
            return None

        volume = np.stack([d[1] for d in dicoms])
        chunks = get_chunk_shape(volume.shape, chunk_size)

        out_dir = Path(out_root) / mrn
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"{accession}.h5"

        if out_file.exists():
            return None

        with h5py.File(out_file, "w") as h5f:

            if compression == "blosc":
                h5f.create_dataset(
                    "volume",
                    data=volume,
                    chunks=chunks,
                    **hdf5plugin.Blosc(cname="zstd", clevel=5),
                )
            elif compression == "gzip":
                h5f.create_dataset("volume", data=volume, chunks=chunks, compression="gzip")
            else:
                h5f.create_dataset("volume", data=volume, chunks=chunks)

            meta = h5f.create_group("metadata")
            slices = meta.create_group("slices")

            for i, (_, _, ds) in enumerate(dicoms):
                s = slices.create_group(str(i))
                try:
                    for elem in ds:
                        try:
                            if elem.value is None:
                                continue
                            s.attrs[safe_key(elem)] = safe_metadata_value(elem.value)
                        except:
                            continue
                except:
                    continue

            h5f.attrs["accession"] = accession
            h5f.attrs["mrn"] = mrn
            h5f.attrs["num_slices"] = volume.shape[0]

            h5f.flush()

        return str(out_file)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -------------------------------------------------
# Batch
# -------------------------------------------------

def convert_all(dicoms_dir, out_root, compression, chunk_size, limit, index, num_jobs):

    tar_files = list(Path(dicoms_dir).rglob("*.tar.gz"))
    subset = tar_files[index::num_jobs]

    print(f"[JOB {index}] processing {len(subset)}")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lock_file, count_file = get_run_state_paths(out_root)

    # initialize shared counter once per run
    with open(lock_file, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if not count_file.exists():
            write_count(count_file, 0)
        fcntl.flock(lockf, fcntl.LOCK_UN)

    written = 0

    for p in tqdm(subset):
        # stop early if global run limit already reached
        if limit is not None:
            with open(lock_file, "a+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                current_total = read_count(count_file)
                if current_total >= limit:
                    print(f"[JOB {index}] global limit reached ({current_total})")
                    fcntl.flock(lockf, fcntl.LOCK_UN)
                    break
                fcntl.flock(lockf, fcntl.LOCK_UN)

        result = convert_accession((p, out_root), compression, chunk_size)

        if result:
            if limit is None:
                written += 1
                continue

            keep_file = False
            with open(lock_file, "a+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                current_total = read_count(count_file)

                if current_total < limit:
                    write_count(count_file, current_total + 1)
                    keep_file = True

                fcntl.flock(lockf, fcntl.LOCK_UN)

            if keep_file:
                written += 1
            else:
                # another job filled the quota first
                try:
                    Path(result).unlink()
                except:
                    pass
                break

    print(f"[JOB {index}] wrote {written}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compression", default="blosc")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--limit_accessions", type=int)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--num_jobs", type=int, default=1)
    return p.parse_args()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    convert_all(
        args.input_dir,
        args.out_dir,
        args.compression,
        args.chunk_size,
        args.limit_accessions,
        args.index,
        args.num_jobs,
    )


