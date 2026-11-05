"""Copy selected LPS experiment data into the Zenodo staging folder.

Per LPS-n folder, only the following are copied:
  - files directly inside LPS-n/
  - LPS-n/CBox/**/*.h5    (including nested subfolders; structure is preserved)
  - LPS-n/CD/*            (files)
  - LPS-n/photo/*.jpg
  - LPS-n/rigaku/*        (XRD exports; only present for some experiments, e.g. LPS-32)
  - LPS-n/xrd_ref/*       (reference XRD patterns, LPS-32)

With --dry-run, the destination folder structure is created but no files are copied.
Existing destination files with the same size are skipped, so the script is re-runnable.
Each file is copied to a .part file and renamed on success; failed copies are retried, and
files that still fail are reported at the end (exit code 1) without stopping the run.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(r"G:\Shared drives\Sutter-Fella Lab\ECRP-Project\Yi-Ru")
DEFAULT_SRC = ROOT / "LPS"
DEFAULT_DST = ROOT / "manuscripts" / "LPS-data-zenodo"
DEFAULT_EXPERIMENTS = [*range(3, 28), 32]

# (subfolder, glob); None subfolder = direct children of LPS-n
RULES = [
    (None, "*"),
    ("CBox", "**/*.h5"),
    ("CD", "*"),
    ("photo", "*.jpg"),
    ("rigaku", "*"),
    ("xrd_ref", "*"),
]


def collect(exp_dir: Path):
    """Yield (relative_path, source_file) for every file to copy in one experiment."""
    for sub, pattern in RULES:
        folder = exp_dir if sub is None else exp_dir / sub
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob(pattern)):
            if f.is_file():
                yield f.relative_to(exp_dir), f


def copy_with_retry(src: Path, dst: Path, retries: int, delay: float):
    """Copy via a .part file and rename on success, so a failed copy never leaves a truncated dst."""
    part = dst.with_name(dst.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            shutil.copy2(src, part)
            if part.stat().st_size != src.stat().st_size:
                raise OSError("size mismatch after copy")
            os.replace(part, dst)
            return
        except OSError as e:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            tqdm.write(f"[RETRY {attempt}/{retries - 1}] {src.name}: {e}")
            time.sleep(delay * attempt)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="n values for LPS-n",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=4,
        help="attempts per file before giving up on it",
    )
    ap.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="base seconds to wait between attempts (scales with attempt)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="create folder structure only; do not copy files",
    )
    args = ap.parse_args()

    plan = []  # (name, exp_dst, [(rel, src_file), ...])
    for n in tqdm(args.experiments, desc="Scanning", unit="exp"):
        name = f"LPS-{n}"
        exp_src = args.src / name
        if not exp_src.is_dir():
            tqdm.write(f"[WARN] missing source folder: {exp_src}")
            continue
        files = list(collect(exp_src))
        exp_dst = args.dst / name
        exp_dst.mkdir(parents=True, exist_ok=True)
        for sub in {rel.parent for rel, _ in files}:
            (exp_dst / sub).mkdir(parents=True, exist_ok=True)
        plan.append((name, exp_dst, files))

    total_files = sum(len(files) for _, _, files in plan)
    total_copied = total_skipped = 0
    failures = []
    if args.dry_run:
        for name, _, files in plan:
            print(f"{name}: {len(files)} files selected")
    else:
        total_bytes = sum(f.stat().st_size for _, _, files in plan for _, f in files)
        with tqdm(
            total=total_bytes,
            desc="Copying",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for name, exp_dst, files in plan:
                bar.set_postfix_str(name)
                for rel, src_file in files:
                    size = src_file.stat().st_size
                    dst_file = exp_dst / rel
                    if dst_file.exists() and dst_file.stat().st_size == size:
                        total_skipped += 1
                    else:
                        try:
                            copy_with_retry(
                                src_file, dst_file, args.retries, args.retry_delay
                            )
                            total_copied += 1
                        except OSError as e:
                            failures.append((src_file, e))
                            tqdm.write(f"[FAIL] {src_file}: {e}")
                    bar.update(size)

    mode = "DRY RUN (folders only)" if args.dry_run else "COPY"
    print(
        f"\n{mode}: {total_files} files selected, {total_copied} copied, {total_skipped} skipped, {len(failures)} failed"
    )
    if failures:
        print("Failed files (re-run the script to retry only these):")
        for f, e in failures:
            print(f"  {f}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
