"""
Fun-Audio-Chat (FAC) Plugin for Hermes Agent
=============================================

Registers ``fac`` TTS and ``fac`` transcription providers that bridge
Alibaba Fun-Audio-Chat 8B as a local voice backend. Works with all
gateway clients — Telegram, Discord, CLI voice mode, etc.

Usage
-----
1. Ensure a FAC server is running at ``ws://127.0.0.1:11236/chat``.
2. Enable the plugin in ``~/.hermes/config.yaml``::

       plugins:
         enabled:
           - fac-s2s

3. Select the provider in config::

       tts:
         provider: fac

       stt:
         provider: fac

Environment variables
---------------------
``FAC_WS_URL``   WebSocket URL (default: ``ws://127.0.0.1:11236/chat``)
``FAC_TIMEOUT``  Response timeout in seconds (default: ``30``)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.tts_provider import TTSProvider
from agent.transcription_provider import TranscriptionProvider

from .fac_bridge import FACClient, FACConnectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plugin metadata (also declared in plugin.yaml)
# ---------------------------------------------------------------------------

__plugin_name__ = "fac-s2s"
__version__ = "1.0.0"
__description__ = (
    "Fun-Audio-Chat (FAC) server-side voice provider — "
    "Alibaba Fun-Audio-Chat 8B TTS + STT backend"
)

# ---------------------------------------------------------------------------
# TTS Provider
# ---------------------------------------------------------------------------


class FACTTSProvider(TTSProvider):
    """TTS provider backed by a local Fun-Audio-Chat server.

    Connects to the FAC WebSocket, sends text, and receives Opus-encoded
    24 kHz audio.
    """

    @property
    def name(self) -> str:
        return "fac"

    @property
    def display_name(self) -> str:
        return "Fun-Audio-Chat TTS"

    def is_available(self) -> bool:
        """Return True when the FAC server is reachable."""
        try:
            client = FACClient()
            return client.is_server_reachable()
        except Exception:
            return False

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return available voices.

        FAC uses CosyVoice3 0.5B for TTS voice quality. The voice
        selection depends on the FAC server configuration. We expose a
        default here — the server may support additional voices.
        """
        return [
            {
                "id": "default",
                "display": "FAC default voice",
                "language": "zh-CN",
                "gender": "female",
            },
            {
                "id": "cosyvoice3",
                "display": "CosyVoice3 0.5B",
                "language": "zh-CN",
                "gender": "female",
            },
        ]

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "fun-audio-chat-8b",
                "display": "Fun-Audio-Chat 8B",
                "languages": ["zh", "en"],
            },
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Alibaba Fun-Audio-Chat 8B – local TTS/STT",
            "env_vars": [
                {
                    "key": "FAC_WS_URL",
                    "prompt": "FAC WebSocket URL",
                    "url": "https://github.com/AlibabaResearch/fun-audio-chat",
                },
            ],
        }

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "opus",
        **extra: Any,
    ) -> str:
        """Synthesize text to audio via FAC server.

        Writes Opus-encoded 24 kHz audio to ``output_path``.

        Returns the absolute path to the written file on success.
        Raises on failure — the dispatcher converts exceptions to the
        standard JSON error envelope.
        """
        from .fac_bridge import run_async
        from .emotional_context import EmotionalContext

        # Apply emotional context from SOUL.md + USER.md
        try:
            emo = EmotionalContext()
            conditioned_text = emo.render(text)
            logger.info("FAC TTS: conditioned text -> %r", conditioned_text.splitlines()[0])
        except Exception as exc:
            logger.warning("Failed to construct emotional context: %s. Using raw text.", exc)
            conditioned_text = text

        config: Dict[str, Any] = {}
        if speed is not None:
            config["speed"] = speed
        if voice is not None:
            config["voice"] = voice
        if model is not None:
            config["model"] = model

        client = FACClient()
        try:
            result = run_async(
                client.synthesize_to_file(
                    text=conditioned_text,
                    output_path=output_path,
                    voice=voice,
                    **config,
                )
            )
        except FACConnectionError as exc:
            raise RuntimeError(
                f"FAC TTS unavailable: {exc}. Is the FAC server running "
                f"at {client._url}?"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"FAC TTS synthesis failed: {exc}"
            ) from exc

        return result


# ---------------------------------------------------------------------------
# STT / Transcription Provider
# ---------------------------------------------------------------------------


class FACTranscriptionProvider(TranscriptionProvider):
    """Transcription (STT) provider backed by a local Fun-Audio-Chat server.

    Connects to the FAC WebSocket, sends Opus-encoded 24 kHz audio, and
    receives transcribed text.
    """

    @property
    def name(self) -> str:
        return "fac"

    @property
    def display_name(self) -> str:
        return "Fun-Audio-Chat STT"

    def is_available(self) -> bool:
        """Return True when the FAC server is reachable."""
        try:
            client = FACClient()
            return client.is_server_reachable()
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "fun-audio-chat-8b",
                "display": "Fun-Audio-Chat 8B",
                "languages": ["zh", "en"],
            },
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Alibaba Fun-Audio-Chat 8B – local TTS/STT",
            "env_vars": [
                {
                    "key": "FAC_WS_URL",
                    "prompt": "FAC WebSocket URL",
                    "url": "https://github.com/AlibabaResearch/fun-audio-chat",
                },
            ],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio file using FAC server.

        Args:
            file_path: Absolute path to the audio file (Opus-encoded
                24 kHz preferred).
            model: Model identifier (unused — FAC uses its single model).
            language: Optional language hint (e.g. ``"en"``, ``"zh"``).

        Returns:
            Standard envelope dict with ``success``, ``transcript``,
            ``provider``, and optionally ``error``.
        """
        from .fac_bridge import run_async

        # Validate the file exists
        expanded_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.isfile(expanded_path):
            return {
                "success": False,
                "transcript": "",
                "error": f"Audio file not found: {file_path}",
                "provider": self.name,
            }

        if os.path.getsize(expanded_path) == 0:
            return {
                "success": False,
                "transcript": "",
                "error": f"Audio file is empty: {file_path}",
                "provider": self.name,
            }

        client = FACClient()
        try:
            transcript = run_async(
                client.transcribe_file(
                    file_path=expanded_path,
                    language=language,
                )
            )
        except FACConnectionError as exc:
            return {
                "success": False,
                "transcript": "",
                "error": (
                    f"FAC server unreachable at {client._url}: {exc}. "
                    f"Is the FAC server running?"
                ),
                "provider": self.name,
            }
        except Exception as exc:
            logger.exception("FAC transcription failed")
            return {
                "success": False,
                "transcript": "",
                "error": f"FAC transcription error: {exc}",
                "provider": self.name,
            }

        return {
            "success": True,
            "transcript": transcript,
            "provider": self.name,
        }


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the FAC TTS and transcription providers.

    This function is called by the Hermes plugin loader when the plugin
    is enabled in ``plugins.enabled``.

    If the FAC server is unreachable at load time, the providers still
    register but ``is_available()`` will return ``False``, so the
    provider picker and tool dispatcher can degrade gracefully.
    """
    # Register TTS provider
    tts_provider = FACTTSProvider()
    ctx.register_tts_provider(tts_provider)
    logger.info(
        "Registered FAC TTS provider '%s' (available: %s)",
        tts_provider.name,
        tts_provider.is_available(),
    )

    # Register transcription (STT) provider
    stt_provider = FACTranscriptionProvider()
    ctx.register_transcription_provider(stt_provider)
    logger.info(
        "Registered FAC transcription provider '%s' (available: %s)",
        stt_provider.name,
        stt_provider.is_available(),
    )
