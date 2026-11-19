import re
from http.client import HTTPException
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image, ImageOps

DATA_DIR = Path(__file__).resolve().parent / "verb_data"
IMAGE_DIR = DATA_DIR / "svo_images"
INPUT = DATA_DIR / "svo_verb_4afc.csv"
OUTPUT = DATA_DIR / "svo_verb_4afc_local.csv"
MAX_BYTES = 20 * 1024 * 1024


def filename_word(word):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(word)).strip()


def download_image(url, destination):
    if destination.is_file():
        return

    request = Request(url, headers={"User-Agent": "SVO-Probes-image-download/1.0"})
    with urlopen(request, timeout=10) as response:
        data = response.read(MAX_BYTES + 1)

    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the download size limit")

    # Rename only after a complete JPEG has been written.
    temporary = destination.with_suffix(".jpg.part")
    try:
        with Image.open(BytesIO(data)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.save(temporary, format="JPEG", quality=95)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    trials = pd.read_csv(INPUT)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    path_columns = [f"image_{i}_path" for i in range(4)]
    for column in path_columns:
        trials[column] = ""

    failures = 0
    for row_index, trial in trials.iterrows():
        trial_id = trial["trial_id"]
        target = filename_word(trial["target_lemma"])

        for i, path_column in enumerate(path_columns):
            lemma = filename_word(trial[f"lemma_{i}"])
            destination = IMAGE_DIR / f"{trial_id}_{target}_image{i}_{lemma}.jpg"

            try:
                download_image(trial[f"image_{i}"], destination)
            except (URLError, OSError, HTTPException, ValueError, Image.DecompressionBombError) as error:
                failures += 1
                print(f"Trial {trial_id}, image {i}: {error}")
                continue

            trials.at[row_index, path_column] = destination.relative_to(DATA_DIR).as_posix()

    trials["all_images_local"] = trials[path_columns].ne("").all(axis=1)
    trials.to_csv(OUTPUT, index=False)

    print(f"Trials with all four images: {trials['all_images_local'].sum():,} / {len(trials):,}")
    print(f"Failed image downloads: {failures:,}")
    print(f"Saved local paths to {OUTPUT}")


if __name__ == "__main__":
    main()