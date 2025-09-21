# data_finder.py
from pathlib import Path

def find_data_root(start_dirs):
    """
    Return the directory that contains both 'Training' and 'Test'.
    Checks common layouts and then searches recursively.
    """
    for base in map(Path, start_dirs):
        if not base.exists():
            continue

        # Quick checks for typical layouts
        candidates = [
            base,
            base / "fruits-360",
            base / "Fruits-360",
            base / "fruits-360_dataset",
            base / "fruits-360_dataset" / "fruits-360",
            base / "archive",
            base / "archive" / "fruits-360",
        ]
        for cand in candidates:
            if (cand / "Training").is_dir() and (cand / "Test").is_dir():
                return cand

        # Deep recursive search
        for p in base.rglob("*"):
            if p.is_dir() and (p / "Training").is_dir() and (p / "Test").is_dir():
                return p
    return None

if __name__ == "__main__":
    roots = [
        r"C:\Users\HP\Desktop\fruits-360",
        r"C:\Users\HP\Desktop\yes\fruits-360",
        r"C:\Users\HP\Desktop",
        Path.cwd(),
    ]
    root = find_data_root(roots)
    print("DATA_ROOT:", root)