"""Convert all .m4a recordings under atlas-voice-data into atlas-voice-data-wav."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert .m4a files into a mirrored .wav directory tree."
    )
    parser.add_argument(
        "--input-dir",
        default="atlas-voice-data",
        help="Source directory containing .m4a files.",
    )
    parser.add_argument(
        "--output-dir",
        default="atlas-voice-data-wav",
        help="Destination directory for converted .wav files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .wav files in the output directory.",
    )
    return parser.parse_args()


def iter_m4a_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.m4a") if path.is_file())


def build_output_path(source_file: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = source_file.relative_to(input_dir)
    return output_dir / relative_path.with_suffix(".wav")


def convert_file(source_file: Path, output_file: Path, ffmpeg_exe: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_file),
    ]
    subprocess.run(command, check=True)


def convert_directory(input_dir: Path, output_dir: Path, overwrite: bool) -> tuple[int, int, int]:
    ffmpeg_exe = get_ffmpeg_exe()
    source_files = iter_m4a_files(input_dir)

    converted = 0
    skipped = 0
    failed = 0

    for source_file in source_files:
        output_file = build_output_path(source_file, input_dir, output_dir)

        if output_file.exists() and not overwrite:
            print(f"Skipped: {output_file}")
            skipped += 1
            continue

        try:
            convert_file(source_file, output_file, ffmpeg_exe)
            print(f"Converted: {source_file} -> {output_file}")
            converted += 1
        except subprocess.CalledProcessError as exc:
            print(f"Failed: {source_file} | ffmpeg exit code: {exc.returncode}")
            failed += 1

    return converted, skipped, failed


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted, skipped, failed = convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    print(
        f"Done. converted={converted} skipped={skipped} failed={failed} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
