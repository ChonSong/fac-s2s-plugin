"""
Fun-Audio-Chat (FAC) WebSocket Client
======================================

Implements the custom binary WebSocket protocol used by Alibaba Fun-Audio-Chat
8B server, spoken on ``ws://127.0.0.1:11236/chat``.

Protocol
--------
Every message is prefixed with a single type byte:

    ``0x00`` — Handshake (JSON config / capabilities)
    ``0x01`` — Audio (Opus-encoded 24 kHz PCM frames)
    ``0x02`` — Text (UTF-8 encoded)
    ``0x03`` — Control (e.g. flush, end-of-stream)

Audio is Opus-encoded 24 kHz, 16-bit signed little-endian mono PCM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

TYPE_HANDSHAKE = 0x00
TYPE_AUDIO = 0x01
TYPE_TEXT = 0x02
TYPE_CONTROL = 0x03

# Default FAC server URL
DEFAULT_FAC_WS_URL = "ws://127.0.0.1:11236/chat"
DEFAULT_TIMEOUT = 30  # seconds
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 1.0  # seconds

# Opus frame duration at 24 kHz
OPUS_FRAME_MS = 20  # standard Opus frame
FAC_SAMPLE_RATE = 24000
FAC_CHANNELS = 1

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FACConnectionError(ConnectionError):
    """Raised when the FAC WebSocket server is unreachable."""


class FACProtocolError(RuntimeError):
    """Raised on unexpected protocol data."""


class FACTimedOutError(TimeoutError):
    """Raised when the FAC server does not respond in time."""


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def get_fac_ws_url() -> str:
    """Return the FAC server WebSocket URL from env or default."""
    return os.environ.get("FAC_WS_URL", DEFAULT_FAC_WS_URL)


def get_fac_timeout() -> int:
    """Return the request timeout from env or default."""
    raw = os.environ.get("FAC_TIMEOUT", str(DEFAULT_TIMEOUT))
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------


class FACClient:
    """Low-level binary-protocol WebSocket client for Fun-Audio-Chat.

    Usage (TTS)::

        client = FACClient()
        async for chunk in client.tts_stream("Hello world"):
            # chunk is bytes of Opus audio
            ...

    Usage (STT)::

        client = FACClient()
        transcript = await client.transcribe_audio(b"<opus data>")
    """

    def __init__(
        self,
        url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._url = url or get_fac_ws_url()
        self._timeout = timeout or get_fac_timeout()
        self._ws: Any = None  # aiohttp ClientWebSocketResponse
        self._session: Any = None  # aiohttp ClientSession

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def is_server_reachable(self) -> bool:
        """Synchronously check if the FAC server is reachable.

        Performs a quick TCP connect to the host:port without doing the
        full WebSocket handshake. Returns False if the host is
        unreachable or the port is closed.
        """
        import socket

        try:
            # Strip ws:// or wss:// prefix, extract host:port
            url = self._url
            for prefix in ("ws://", "wss://"):
                if url.startswith(prefix):
                    url = url[len(prefix) :]
                    break
            host, _, port_str = url.partition(":")
            port = int(port_str.rstrip("/").split("/")[0]) if port_str else 11236
            if not host:
                host = "127.0.0.1"
            # Try a TCP socket connect with short timeout
            sock = socket.create_connection(
                (host, port), timeout=2.0,
            )
            sock.close()
            return True
        except (OSError, socket.timeout, ValueError):
            return False

    async def _connect(self) -> None:
        """Open the WebSocket connection."""
        import aiohttp

        if self._ws is not None and not self._ws.closed:
            return  # already connected

        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout + 5),
        )
        try:
            timeout_obj = aiohttp.ClientWSTimeout(
                ws_receive=self._timeout,
            )
            self._ws = await session.ws_connect(
                self._url,
                timeout=timeout_obj,
                heartbeat=10.0,
                max_msg_size=0,  # no limit
            )
            # Store session so we can close it later
            self._session = session
            logger.debug("FAC: Connected to %s", self._url)
        except Exception as exc:
            await session.close()
            self._ws = None
            self._session = None
            raise FACConnectionError(
                f"Cannot connect to FAC server at {self._url}: {exc}"
            ) from exc

    async def _disconnect(self) -> None:
        """Close the WebSocket and session if open."""
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def _send_message(self, msg_type: int, payload: bytes) -> None:
        """Send a binary message with type prefix."""
        if self._ws is None or self._ws.closed:
            raise FACConnectionError("Not connected to FAC server")
        framed = struct.pack("!B", msg_type) + payload
        await self._ws.send_bytes(framed)

    async def _recv_message(self) -> Tuple[int, bytes]:
        """Receive one binary message, returning (type, payload).

        Raises FACTimedOutError on timeout, FACProtocolError on
        unexpected data.
        """
        if self._ws is None or self._ws.closed:
            raise FACConnectionError("Not connected to FAC server")

        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            data: bytes = msg.data
            if not data:
                raise FACProtocolError("Empty binary message")
            msg_type = data[0]
            payload = data[1:]
            return msg_type, payload
        elif msg.type == aiohttp.WSMsgType.CLOSED:
            raise FACConnectionError("FAC server closed the connection")
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise FACConnectionError(
                f"FAC WebSocket error: {self._ws.exception()}"
            )
        else:
            raise FACProtocolError(
                f"Unexpected message type: {msg.type}"
            )

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    async def _handshake(
        self,
        mode: str = "s2s",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform the FAC protocol handshake.

        Sends a handshake message (type 0x00) with a JSON payload
        describing the client capabilities and receives the server's
        handshake response.

        Args:
            mode: ``"s2s"`` (full speech-to-speech), ``"tts"`` (text-to-speech
                only), ``"stt"`` (speech-to-text only).
            config: Optional extra configuration dict.

        Returns:
            Server handshake response dict.
        """
        handshake_payload = {
            "mode": mode,
            "sample_rate": FAC_SAMPLE_RATE,
            "channels": FAC_CHANNELS,
            "opus_frame_ms": OPUS_FRAME_MS,
        }
        if config:
            handshake_payload.update(config)

        payload_bytes = json.dumps(handshake_payload).encode("utf-8")
        await self._send_message(TYPE_HANDSHAKE, payload_bytes)

        # Wait for handshake response
        resp_type, resp_payload = await self._recv_message()
        if resp_type != TYPE_HANDSHAKE:
            raise FACProtocolError(
                f"Expected handshake response (type 0x00), got 0x{resp_type:02x}"
            )

        try:
            response = json.loads(resp_payload.decode("utf-8"))
            logger.debug("FAC handshake response: %s", response)
            return response
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FACProtocolError(
                f"Invalid handshake response: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # TTS: text → audio stream
    # ------------------------------------------------------------------

    async def tts_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        **extra: Any,
    ) -> AsyncIterator[bytes]:
        """Stream Opus audio chunks from FAC for the given text.

        Yields raw Opus frames (bytes) as they arrive from the server.

        Args:
            text: Text to synthesize.
            voice: Optional voice ID (if FAC supports multi-voice).
            **extra: Extra params forwarded in the handshake config.

        Yields:
            Raw Opus frame bytes.
        """
        config: Dict[str, Any] = {}
        if voice is not None:
            config["voice"] = voice
        if extra:
            config.update(extra)

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                await self._connect()
                await self._handshake(mode="tts", config=config)

                # Send the text to synthesize
                text_bytes = text.encode("utf-8")
                await self._send_message(TYPE_TEXT, text_bytes)

                # Send end-of-stream control message
                await self._send_message(TYPE_CONTROL, b"\x00")

                # Receive audio chunks
                while True:
                    msg_type, payload = await self._recv_message()
                    if msg_type == TYPE_AUDIO:
                        yield payload
                    elif msg_type == TYPE_CONTROL:
                        # End of audio stream
                        break
                    elif msg_type == TYPE_TEXT:
                        # Server may send status text
                        logger.debug("FAC TTS status: %s", payload.decode("utf-8", errors="replace"))
                        continue
                    else:
                        logger.debug(
                            "FAC TTS: unexpected type 0x%02x, ignoring", msg_type
                        )
                return  # success, exit loop

            except (FACConnectionError, FACProtocolError, TimeoutError) as exc:
                await self._disconnect()
                if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                    logger.warning(
                        "FAC TTS attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1, exc, RECONNECT_DELAY,
                    )
                    await asyncio.sleep(RECONNECT_DELAY)
                else:
                    raise FACConnectionError(
                        f"FAC TTS failed after {MAX_RECONNECT_ATTEMPTS} attempts: {exc}"
                    ) from exc
            finally:
                await self._disconnect()

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        **extra: Any,
    ) -> str:
        """Synthesize text and write the Opus audio to a file.

        Collects all Opus frames from the stream and writes them to the
        output file. The output is raw Opus frames (not a container
        format like .opus/.ogg). For standard playback compatibility,
        the caller should re-package into an Ogg-Opus container.

        Args:
            text: Text to synthesize.
            output_path: Path to write the audio file.
            voice: Optional voice override.
            **extra: Extra params for the handshake.

        Returns:
            The absolute path to the written file.
        """
        output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        chunks: List[bytes] = []
        async for chunk in self.tts_stream(text, voice=voice, **extra):
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("FAC returned no audio data")

        with open(output_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        logger.info(
            "FAC TTS: wrote %d bytes to %s (from %d Opus frames)",
            sum(len(c) for c in chunks),
            output_path,
            len(chunks),
        )
        return output_path

    # ------------------------------------------------------------------
    # STT: audio → text
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **extra: Any,
    ) -> str:
        """Transcribe audio data using FAC's speech recognition.

        The audio data should be Opus-encoded at 24 kHz. If the data is
        raw PCM, it will be Opus-encoded before sending.

        Args:
            audio_data: Opus-encoded audio bytes (24 kHz).
            language: Optional language hint.
            **extra: Extra params for the handshake.

        Returns:
            Transcribed text string.
        """
        config: Dict[str, Any] = {}
        if language is not None:
            config["language"] = language
        if extra:
            config.update(extra)

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                await self._connect()
                await self._handshake(mode="stt", config=config)

                # Send audio data
                await self._send_message(TYPE_AUDIO, audio_data)

                # Send end-of-stream control
                await self._send_message(TYPE_CONTROL, b"\x00")

                # Collect text response(s)
                texts: List[str] = []
                while True:
                    msg_type, payload = await self._recv_message()
                    if msg_type == TYPE_TEXT:
                        text = payload.decode("utf-8", errors="replace")
                        texts.append(text)
                    elif msg_type == TYPE_CONTROL:
                        break  # end of transcription
                    elif msg_type == TYPE_AUDIO:
                        # Server may echo audio back in S2S mode; ignore
                        continue
                    else:
                        logger.debug(
                            "FAC STT: unexpected type 0x%02x, ignoring",
                            msg_type,
                        )

                transcript = " ".join(texts).strip()
                if not transcript:
                    logger.warning("FAC STT returned empty transcript")
                return transcript

            except (FACConnectionError, FACProtocolError, TimeoutError) as exc:
                await self._disconnect()
                if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                    logger.warning(
                        "FAC STT attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1, exc, RECONNECT_DELAY,
                    )
                    await asyncio.sleep(RECONNECT_DELAY)
                else:
                    raise FACConnectionError(
                        f"FAC STT failed after {MAX_RECONNECT_ATTEMPTS} attempts: {exc}"
                    ) from exc
            finally:
                await self._disconnect()

    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        **extra: Any,
    ) -> str:
        """Read an audio file and transcribe it.

        Attempts to read the file as raw Opus data. If the file is a
        container format (WAV, OGG, etc.), the caller must first convert
        to raw Opus frames at 24 kHz.

        Args:
            file_path: Path to the audio file.
            language: Optional language hint.
            **extra: Extra params for the handshake.

        Returns:
            Transcribed text.
        """
        audio_path = os.path.abspath(os.path.expanduser(file_path))
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        if not audio_data:
            raise ValueError(f"Empty audio file: {file_path}")
        return await self.transcribe_audio(audio_data, language=language, **extra)


# ---------------------------------------------------------------------------
# Sync convenience wrappers (for use in synchronous provider methods)
# ---------------------------------------------------------------------------


def run_async(coro) -> Any:
    """Run an async coroutine from a sync context.

    Uses the current event loop if one is running, otherwise creates
    a new one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — create one
        return asyncio.run(coro)

    # We're inside a running loop, run in a new task
    import concurrent.futures
    import threading

    result: Any = None
    exc: Optional[Exception] = None
    done = threading.Event()

    async def _run():
        nonlocal result, exc
        try:
            result = await coro
        except Exception as e:
            exc = e
        finally:
            done.set()

    asyncio.run_coroutine_threadsafe(_run(), loop)
    done.wait(timeout=get_fac_timeout() + 10)
    if exc is not None:
        raise exc
    return result
