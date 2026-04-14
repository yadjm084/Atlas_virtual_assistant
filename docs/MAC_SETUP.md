# Mac Setup

From the project root on your Mac:

```bash
chmod +x scripts/setup_mac.sh
./scripts/setup_mac.sh
```

If macOS opens an `Xcode Command Line Tools` installer window, finish that install first and then run `./scripts/setup_mac.sh` again.

When setup finishes:

```bash
source .venv/bin/activate
python app.py
```

Optional checks:

```bash
python demo_smoke_test.py --skip-tts
```

Optional movie API environment variables:

```bash
export TMDB_API_KEY="your_api_key"
export TMDB_BEARER_TOKEN="your_bearer_token"
```

Notes:

- The script is macOS-only.
- It expects to run from this repo and creates a local `.venv`.
- It installs Homebrew if missing, then uses Homebrew for `python@3.10` and `ffmpeg`.
- The app can still download the Whisper `tiny` model on first run.
