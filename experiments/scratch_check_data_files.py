from pathlib import Path

data_dir = Path("data")
for f in data_dir.iterdir():
    if f.is_file():
        print(f"File: {f.name}, Size: {f.stat().st_size / 1024 / 1024:.2f} MB")
