from pathlib import Path  # Provides an object-oriented interface for file paths.


# File extensions accepted by the dataset loader (case-insensitive).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_directory(directory: Path) -> bool:
    # Confirm that the supplied path both exists and refers to a directory.
    return directory.exists() and directory.is_dir()


def is_image_file(file_path: Path) -> bool:
    # Normalize the extension before checking it against the supported formats.
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def sort_image_files(image_files: list[Path]) -> list[Path]:
    # Sort numbered filenames (for example, 2.png before 10.png) numerically.
    return sorted(image_files, key=lambda file: int(file.stem))


def list_image_files(directory: Path) -> list[Path]:
    # Inspect only direct children; nested directories are not included.
    image_files = [
        file
        for file in directory.iterdir()
        # Keep regular files whose extension is one of the supported image types.
        if file.is_file() and is_image_file(file)
    ]

    # Return files in their numeric filename order for deterministic processing.
    return sort_image_files(image_files)
