from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def load_model(model_path: str) -> Any:
    try:
        from mlx_audio.stt.generate import load_model as mlx_load_model
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is required on macOS Apple Silicon. Install transclip[mlx]."
        ) from exc

    return mlx_load_model(model_path)


def generate_transcription(model: Any, audio_path: Path, output_stem: str) -> Any:
    try:
        from mlx_audio.stt.generate import generate_transcription as mlx_generate
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is required on macOS Apple Silicon. Install transclip[mlx]."
        ) from exc

    kwargs = {
        "output_path": output_stem,
        "format": "txt",
    }
    signature = inspect.signature(mlx_generate)
    if "model" in signature.parameters:
        kwargs["model"] = model
        kwargs["audio"] = str(audio_path)
    else:
        if not isinstance(model, str):
            raise RuntimeError("installed mlx-audio generate_transcription does not accept a loaded model")
        kwargs["model_path"] = model
        kwargs["audio_path"] = str(audio_path)
    return mlx_generate(**kwargs)
