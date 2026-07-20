# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from pathlib import Path

from datasets import Image as DatasetsImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "images"
DATASET_NAME = "microsoft/cats_vs_dogs"
DATASET_SPLIT = "train"
NUM_IMAGES = 1024
MIN_IMAGE_SIZE = 224
OUTPUT_PREFIX = "cats_vs_dogs"
LEGACY_IMAGE_GLOBS = ("tiny_imagenet_*.JPEG",)


def get_image_bytes(image):
    if image.get("bytes") is not None:
        return image["bytes"]

    image_path = image.get("path")
    if image_path is None:
        raise ValueError("Dataset image sample does not contain bytes or a path")
    return Path(image_path).read_bytes()


def write_verified_jpeg(image, output_path):
    image_bytes = get_image_bytes(image)
    with Image.open(BytesIO(image_bytes)) as source_image:
        if source_image.format != "JPEG":
            raise ValueError(f"Expected JPEG source image, got {source_image.format}")
        if min(source_image.size) < MIN_IMAGE_SIZE:
            return False
    output_path.write_bytes(image_bytes)
    return True


def download_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    dataset = dataset.cast_column("image", DatasetsImage(decode=False))
    downloaded = 0
    iterator = iter(dataset)

    try:
        with tqdm(total=NUM_IMAGES, desc=f"Downloading {DATASET_NAME}/{DATASET_SPLIT}") as progress:
            for sample in iterator:
                output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{DATASET_SPLIT}_{downloaded:06d}.JPEG"
                if write_verified_jpeg(sample["image"], output_path):
                    downloaded += 1
                    progress.update()
                    if downloaded == NUM_IMAGES:
                        break
    finally:
        close_iterator = getattr(iterator, "close", None)
        if close_iterator is not None:
            close_iterator()

    if downloaded != NUM_IMAGES:
        raise RuntimeError(f"Expected to download {NUM_IMAGES} JPEG images, downloaded {downloaded}")

    for legacy_glob in LEGACY_IMAGE_GLOBS:
        for legacy_path in OUTPUT_DIR.glob(legacy_glob):
            legacy_path.unlink()

    print(f"Downloaded {downloaded} JPEG images to {OUTPUT_DIR}")


def main():
    download_images()


if __name__ == "__main__":
    main()
