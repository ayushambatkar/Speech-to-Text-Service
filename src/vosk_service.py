from __future__ import annotations

import json
from dataclasses import dataclass
import asyncio
from typing import List, Tuple

import numpy as np
from vosk import Model, KaldiRecognizer

from .speech_to_text_service import SpeechToTextService
from .transcribe_interface import TranscribeServiceInterface


@dataclass
class SimpleTranscriptionInfo:
    language: str = "unknown"
    language_probability: float = 1.0
    duration: float = 0.0


class VoskTranscribeService(TranscribeServiceInterface):
    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        self.model = Model(model_path)
        self.sample_rate = sample_rate

    def _audio_to_pcm16(self, audio: np.ndarray) -> bytes:
        if audio.dtype != np.int16:
            # assume float32 in range [-1, 1]
            arr = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        else:
            arr = audio
        return arr.tobytes()

    def transcribe(self, audio, language: str | None = None, task: str = "transcribe", word_timestamps: bool = False, beam_size: int = 5, vad_filter: bool = True) -> Tuple[List[dict], SimpleTranscriptionInfo]:
        # Accept pre-decoded numpy arrays or bytes-like; decode if needed
        if isinstance(audio, (bytes, bytearray)):
            audio_arr = SpeechToTextService.decode_audio(bytes(audio), sample_rate=self.sample_rate)
        else:
            audio_arr = audio

        pcm = self._audio_to_pcm16(audio_arr)
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)

        # Feed audio in chunks
        chunk_size = 4000
        for i in range(0, len(pcm), chunk_size):
            rec.AcceptWaveform(pcm[i : i + chunk_size])

        final = rec.FinalResult()
        try:
            parsed = json.loads(final)
            text = parsed.get("text", "").strip()
            words = parsed.get("result", [])
        except Exception:
            text = ""
            words = []

        duration = float(len(audio_arr) / float(self.sample_rate)) if hasattr(audio_arr, "__len__") else 0.0
        info = SimpleTranscriptionInfo(language=(language or "unknown"), language_probability=1.0, duration=duration)

        segment = {
            "start": 0.0,
            "end": round(duration, 3),
            "text": text,
        }
        if word_timestamps and words:
            segment["words"] = [
                {"word": w.get("word", ""), "start": round(w.get("start", 0.0), 3), "end": round(w.get("end", 0.0), 3)}
                for w in words
            ]

        return [segment], info

    async def transcribe_to_queue(self, audio, queue: asyncio.Queue, language: str | None = None, task: str = "transcribe", word_timestamps: bool = False) -> None:
        loop = asyncio.get_running_loop()

        def _run():
            try:
                segments, info = self.transcribe(audio, language=language, task=task, word_timestamps=word_timestamps)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "info", "language": info.language, "language_probability": info.language_probability, "duration": round(info.duration, 3)}),
                    loop,
                ).result()

                for seg in segments:
                    payload = {"type": "segment", **seg}
                    asyncio.run_coroutine_threadsafe(queue.put(payload), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "message": str(exc)}), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        await loop.run_in_executor(None, _run)

    def transcribe_partial(self, audio, language: str | None = None, word_timestamps: bool = False, beam_size: int = 2, vad_filter: bool = False) -> tuple[list[dict], SimpleTranscriptionInfo]:
        # For Vosk, partial and full behave similarly for a given snapshot.
        return self.transcribe(audio, language=language, word_timestamps=word_timestamps, beam_size=beam_size, vad_filter=vad_filter)
