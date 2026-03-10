import io
import asyncio
from typing import AsyncIterator, Iterator, Optional, Tuple

import av
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo, TranscriptionOptions
from .transcribe_interface import TranscribeServiceInterface


class SpeechToTextService(TranscribeServiceInterface):
    """
    Wraps faster-whisper for synchronous and async streaming transcription.
    Intended to be instantiated once (module-level) and reused across requests.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode_audio(audio_bytes: bytes, sample_rate: int = 16_000) -> np.ndarray:
        """
        Decode arbitrary audio bytes (wav, mp3, webm, ogg, mp4, …) into a
        mono float32 numpy array resampled to `sample_rate` Hz.
        Uses PyAV - no external ffmpeg binary required.
        """
        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=sample_rate,
        )
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray()[0])

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: "str | np.ndarray",
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Tuple[Iterator, TranscriptionInfo]:
        """
        Thin wrapper around WhisperModel.transcribe.

        `audio` can be:
          - a file path (str/Path)
          - a pre-decoded float32 numpy array (shape: [samples])

        Returns (segments_generator, info).  Iterating segments_generator is
        CPU-bound - always do it inside a thread / executor.
        """
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )
        return segments, info

    # ------------------------------------------------------------------
    # Async streaming helper (feeds an asyncio.Queue for SSE / WS)
    # ------------------------------------------------------------------

    async def transcribe_to_queue(
        self,
        audio: "str | np.ndarray",
        queue: asyncio.Queue,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
    ) -> None:
        """
        Runs transcription in a thread pool and pushes each segment dict
        into *queue* as soon as it is ready.

        Sentinel value ``None`` is pushed when transcription is finished.
        On error, an ``{"error": str}`` dict is pushed before None.
        """
        loop = asyncio.get_running_loop()

        def _run() -> None:
            try:
                segments, info = self.transcribe(
                    audio,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                )
                # Push metadata first
                asyncio.run_coroutine_threadsafe(
                    queue.put(
                        {
                            "type": "info",
                            "language": info.language,
                            "language_probability": round(info.language_probability, 4),
                            "duration": round(info.duration, 3),
                        }
                    ),
                    loop,
                ).result()

                for segment in segments:
                    payload: dict = {
                        "type": "segment",
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "text": segment.text.strip(),
                    }
                    if word_timestamps and segment.words:
                        payload["words"] = [
                            {
                                "word": w.word,
                                "start": round(w.start, 3),
                                "end": round(w.end, 3),
                                "probability": round(w.probability, 3),
                            }
                            for w in segment.words
                        ]
                    asyncio.run_coroutine_threadsafe(queue.put(payload), loop).result()

            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "message": str(exc)}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        await loop.run_in_executor(None, _run)

    # ------------------------------------------------------------------
    # Synchronous helper (for run_in_executor partial transcription)
    # ------------------------------------------------------------------

    def transcribe_sync(
        self,
        audio: "str | np.ndarray",
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> tuple[list[dict], object]:
        """
        Runs transcription fully in the calling thread and returns
        ``(segment_dicts, info)`` — suitable for use inside run_in_executor.
        """
        segments_iter, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )
        return (
            [SpeechToTextService.segment_to_dict(s, word_timestamps) for s in segments_iter],
            info,
        )

    def transcribe_partial(
        self,
        audio: "str | np.ndarray",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        beam_size: int = 2,
        vad_filter: bool = False,
    ) -> tuple[list[dict], object]:
        """Run a low-latency partial transcription pass.

        This wraps `transcribe_sync` but exposes a clearer API for streaming
        partials (returns list[dict], info).
        """
        return self.transcribe_sync(
            audio,
            language=language,
            task="transcribe",
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def segment_to_dict(segment: Segment, word_timestamps: bool = False) -> dict:
        payload = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        }
        if word_timestamps and segment.words:
            payload["words"] = [
                {
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 3),
                }
                for w in segment.words
            ]
        return payload
