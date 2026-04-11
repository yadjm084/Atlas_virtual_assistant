from pathlib import Path
from pydub import AudioSegment

# Folder containing your dataset
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "user_verification"

# Go through all .m4a files recursively
for m4a_file in DATASET_DIR.rglob("*.m4a"):
    wav_file = m4a_file.with_suffix(".wav")

    try:
        audio = AudioSegment.from_file(m4a_file, format="m4a")
        audio.export(wav_file, format="wav")
        print(f"Converted: {m4a_file.name} -> {wav_file.name}")
    except Exception as e:
        print(f"Failed: {m4a_file} | Error: {e}")