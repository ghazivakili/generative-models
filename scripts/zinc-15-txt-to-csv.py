from pathlib import Path

import tqdm
import pandas as pd


source_dir = Path("/root/tranches").resolve()
target_dir = source_dir

txt_files: list[Path] = list(source_dir.glob("*.txt"))


print(f"Found {len(txt_files)} text files. Here's the first one: {txt_files[0]}")

file: Path
for file in tqdm.tqdm(txt_files):
    df = pd.read_csv(str(file), delimiter="\t")
    target_path = file.parent / f"{file.stem}.csv"
    df.to_csv(str(target_path), index=False)
    file.unlink()
