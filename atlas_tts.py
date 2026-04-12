"""Text-to-speech helpers for the Atlas demo."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

try:
    import edge_tts
except Exception as e:  # pragma: no cover - import status only
    edge_tts = None
    EDGE_TTS_STATUS = f"edge-tts unavailable: {e}"
else:
    EDGE_TTS_STATUS = "edge-tts available"


DEFAULT_TTS_VOICE = "en-CA-ClaraNeural"
DEFAULT_TTS_RATE = "+0%"
DEFAULT_TTS_PITCH = "+0Hz"
DEFAULT_TTS_VOLUME = "+0%"


async def _save_edge_tts(text: str, output_path: Path):
    communicate = edge_tts.Communicate(
        text=text,
        voice=DEFAULT_TTS_VOICE,
        rate=DEFAULT_TTS_RATE,
        pitch=DEFAULT_TTS_PITCH,
        volume=DEFAULT_TTS_VOLUME,
    )
    await communicate.save(str(output_path))


def synthesize_tts_audio(text: str, output_dir: Path) -> Path:
    if edge_tts is None:
        raise RuntimeError(EDGE_TTS_STATUS)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"atlas_tts_{uuid4().hex}.mp3"
    asyncio.run(_save_edge_tts(text, output_path))
    return output_path
