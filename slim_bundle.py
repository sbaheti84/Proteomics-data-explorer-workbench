"""
Remove files the packaged app never uses, so the download has far fewer files
and unzips much faster. Run after PyInstaller, on the folder that holds the
bundled libraries:

    python slim_bundle.py dist/ProteomicsWorkbench/_internal

What is removed (none of it is loaded when the app runs):
  - folders named "tests" (test suites of numpy, pandas, scipy, statsmodels,
    scikit-learn, pyarrow, ... and of R packages)
  - *.pyi type-hint files (only used by code editors)
  - pyarrow's C++ headers and sources (only used to compile code against pyarrow)

Prints how many files each top-level folder has afterwards, so a build log
shows where the file count comes from.
"""
import os
import shutil
import sys

REMOVE_DIR_NAMES = {"tests"}
REMOVE_SUFFIXES = (".pyi",)
REMOVE_PATHS = ("pyarrow/include", "pyarrow/includes", "pyarrow/src")


def count_files(path: str) -> int:
    return sum(len(files) for _, _, files in os.walk(path))


def slim(root: str) -> None:
    before = count_files(root)
    removed_dirs = 0
    removed_files = 0

    for rel in REMOVE_PATHS:
        p = os.path.join(root, *rel.split("/"))
        if os.path.isdir(p) and not os.path.islink(p):
            removed_files += count_files(p)
            shutil.rmtree(p)
            removed_dirs += 1

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for d in list(dirnames):
            full = os.path.join(dirpath, d)
            if d in REMOVE_DIR_NAMES and not os.path.islink(full):
                removed_files += count_files(full)
                shutil.rmtree(full)
                removed_dirs += 1
                dirnames.remove(d)  # do not walk into a deleted folder
        for f in filenames:
            if f.endswith(REMOVE_SUFFIXES):
                full = os.path.join(dirpath, f)
                if not os.path.islink(full):
                    os.remove(full)
                    removed_files += 1

    after = count_files(root)
    print(f"slim_bundle: {before:,} files -> {after:,} files "
          f"(removed {removed_files:,} files in {removed_dirs:,} folders)")

    print("Largest folders by file count:")
    rows = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and not os.path.islink(p):
            rows.append((count_files(p), name))
    for n, name in sorted(rows, reverse=True)[:15]:
        print(f"  {n:>7,}  {name}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit("Usage: python slim_bundle.py <bundle folder>")
    slim(sys.argv[1])