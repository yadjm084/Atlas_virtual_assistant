"""Convert .m4a files to .wav recursively for a given directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory to scan recursively for .m4a files.",
    )
    return parser.parse_args()


def convert_directory(input_dir: Path):
    converted = 0
    failed = 0

    for m4a_file in input_dir.rglob("*.m4a"):
        wav_file = m4a_file.with_suffix(".wav")

        try:
            audio = AudioSegment.from_file(m4a_file, format="m4a")
            audio.export(wav_file, format="wav")
            print(f"Converted: {m4a_file} -> {wav_file}")
            converted += 1
        except Exception as e:
            print(f"Failed: {m4a_file} | Error: {e}")
            failed += 1

    print(f"Done. converted={converted} failed={failed}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    convert_directory(input_dir)


if __name__ == "__main__":
    main()
