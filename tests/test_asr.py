from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.asr import ASRSegment, LocalTranscriber, Transcription
from scripts.evaluate_auto_asr_pilot import _edit_distance, _words


class LocalTranscriberTests(unittest.TestCase):
    def test_word_error_count_normalizes_case_and_punctuation(self) -> None:
        self.assertEqual(_edit_distance(_words("Привет, мир!"), _words("привет мир")), 0)
        self.assertEqual(_edit_distance(_words("как дела"), _words("как там дела")), 1)

    def test_span_uses_only_overlapping_asr_segments(self) -> None:
        result = Transcription(
            text="первая вторая",
            segments=(ASRSegment(0.0, 2.0, "первая"), ASRSegment(5.0, 7.0, "вторая")),
        )
        self.assertEqual(result.text_for_span(0.0, 5.0), "первая")
        self.assertEqual(result.text_for_span(5.0, 10.0), "вторая")
        crossing = Transcription("фраза", (ASRSegment(0.0, 5.7, "фраза"),))
        self.assertEqual(crossing.text_for_span(0.0, 5.0), "фраза")
        self.assertEqual(crossing.text_for_span(5.0, 10.0), "")

    def test_model_runs_lazily_and_result_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "speech.wav"
            audio.write_bytes(b"placeholder")
            transcriber = LocalTranscriber(Path(tmp) / "cache")
            model = SimpleNamespace(transcribe=lambda *args, **kwargs: (
                iter([SimpleNamespace(start=0.0, end=1.2, text=" Привет мир ")]), None
            ))
            with patch.object(transcriber, "_ensure_model", return_value=model) as load:
                first = transcriber.transcribe_file(audio)
                second = transcriber.transcribe_file(audio)
            self.assertEqual(first.text, "Привет мир")
            self.assertEqual(first, second)
            load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
