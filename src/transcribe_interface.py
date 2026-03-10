from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import asyncio


class TranscribeServiceInterface(ABC):
    @abstractmethod
    def transcribe(self, audio, language: str | None = None, task: str = "transcribe", word_timestamps: bool = False, beam_size: int = 5, vad_filter: bool = True) -> Tuple[Iterator, object]:
        """Synchronous/full transcription call.

        Returns a pair (segments_iterable, info).
        The concrete implementation may return Segment-like objects or dicts
        for each segment; callers should handle both.
        """

    @abstractmethod
    async def transcribe_to_queue(self, audio, queue: asyncio.Queue, language: str | None = None, task: str = "transcribe", word_timestamps: bool = False) -> None:
        """Run transcription and push segment dicts / metadata into *queue*.

        Implementations should push a metadata/info payload first, then
        individual segment payloads (dicts). Push ``None`` as sentinel when
        finished.
        """

    @abstractmethod
    def transcribe_partial(self, audio, language: str | None = None, word_timestamps: bool = False, beam_size: int = 2, vad_filter: bool = False) -> tuple[list[dict], object]:
        """Run a low-latency partial transcription pass on the provided audio.

        Returns a (list_of_segment_dicts, info) pair suitable for streaming
        / partial-update scenarios.
        """
