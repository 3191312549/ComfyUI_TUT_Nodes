import unittest
import wave
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from ComfyUI_TUT_Nodes.categories import AUDIO
from ComfyUI_TUT_Nodes.nodes.audio.load import (
    TUT_AdvancedAudioLoader,
    _decode_audio_file,
    _file_fingerprint,
)


MODULE = "ComfyUI_TUT_Nodes.nodes.audio.load"


class AdvancedAudioLoaderTests(unittest.TestCase):
    def _run(self, waveform, sample_rate=10, start_time=0.0, end_time=0.0):
        with patch(f"{MODULE}._resolve_audio_path", return_value="test.wav"), patch(
            f"{MODULE}._decode_audio_file", return_value=(waveform, sample_rate)
        ):
            return TUT_AdvancedAudioLoader().load_audio(
                "test.wav", start_time=start_time, end_time=end_time
            )

    def test_public_contract_and_upload_input(self):
        with patch(f"{MODULE}._audio_input_files", return_value=["demo.wav"]):
            inputs = TUT_AdvancedAudioLoader.INPUT_TYPES()["required"]
        self.assertEqual(list(inputs), ["audio_file", "start_time", "end_time"])
        self.assertTrue(inputs["audio_file"][1]["audio_upload"])
        self.assertEqual(inputs["start_time"][1]["default"], 0.0)
        self.assertEqual(inputs["end_time"][1]["default"], 0.0)
        self.assertEqual(TUT_AdvancedAudioLoader.CATEGORY, AUDIO)
        self.assertEqual(
            TUT_AdvancedAudioLoader.RETURN_TYPES,
            ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "INT", "INT"),
        )
        self.assertEqual(
            TUT_AdvancedAudioLoader.RETURN_NAMES,
            ("audio", "duration", "start_time", "end_time", "sample_rate", "channels"),
        )

    def test_frontend_editor_and_chinese_labels_are_registered(self):
        project_root = Path(__file__).resolve().parents[1]
        editor = (project_root / "js" / "tut_audio_loader.js").read_text(encoding="utf-8")
        labels = (project_root / "js" / "tut_chinese_ui.js").read_text(encoding="utf-8")
        for marker in (
            "TUT_AdvancedAudioLoader",
            "const MIN_WIDTH = 520",
            "播放",
            "停止",
            "重置全长",
            'return "seek"',
            'state.drag === "seek"',
            "state.audio.currentTime = clamp(time, start, end)",
            "pointerdown",
            "pointerleave",
        ):
            self.assertIn(marker, editor)
        for marker in ("audio_file: \"音频文件\"", "sample_rate: \"采样率\"", "channels: \"声道数\""):
            self.assertIn(marker, labels)

    def test_end_zero_selects_complete_mono_audio(self):
        source = torch.arange(20, dtype=torch.float32).reshape(1, 20)
        audio, duration, start, end, sample_rate, channels = self._run(source)
        self.assertEqual(audio["waveform"].shape, (1, 1, 20))
        self.assertTrue(torch.equal(audio["waveform"][0], source))
        self.assertEqual((duration, start, end, sample_rate, channels), (2.0, 0.0, 2.0, 10, 1))

    def test_middle_trim_is_sample_aligned_and_preserves_channels(self):
        source = torch.arange(60, dtype=torch.float32).reshape(3, 20)
        audio, duration, start, end, sample_rate, channels = self._run(
            source, start_time=0.26, end_time=1.24
        )
        self.assertEqual(audio["waveform"].shape, (1, 3, 9))
        self.assertTrue(torch.equal(audio["waveform"][0], source[:, 3:12]))
        self.assertAlmostEqual(duration, 0.9)
        self.assertAlmostEqual(start, 0.3)
        self.assertAlmostEqual(end, 1.2)
        self.assertEqual((sample_rate, channels), (10, 3))

    def test_end_is_clamped_to_audio_length(self):
        source = torch.zeros((2, 20), dtype=torch.float32)
        audio, duration, start, end, _, channels = self._run(
            source, start_time=1.0, end_time=99.0
        )
        self.assertEqual(audio["waveform"].shape, (1, 2, 10))
        self.assertEqual((duration, start, end, channels), (1.0, 1.0, 2.0, 2))

    def test_invalid_and_empty_ranges_raise_chinese_errors(self):
        cases = [
            (torch.zeros((1, 20)), 2.0, 0.0, "裁剪范围无效"),
            (torch.zeros((1, 20)), 1.0, 1.0, "裁剪范围无效"),
            (torch.zeros((1, 0)), 0.0, 0.0, "音频为空"),
        ]
        for waveform, start, end, message in cases:
            with self.subTest(start=start, end=end, shape=tuple(waveform.shape)):
                with self.assertRaisesRegex(ValueError, message):
                    self._run(waveform, start_time=start, end_time=end)

    def test_decode_error_mentions_selected_filename(self):
        with patch(f"{MODULE}._resolve_audio_path", return_value="folder/broken.wav"), patch(
            f"{MODULE}._decode_audio_file", side_effect=ValueError("文件中没有可用的音轨。")
        ):
            with self.assertRaisesRegex(ValueError, "broken.wav.*没有可用的音轨"):
                TUT_AdvancedAudioLoader().load_audio("broken.wav")

    def test_content_fingerprint_changes(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sound.bin"
            path.write_bytes(b"first")
            first = _file_fingerprint(str(path))
            path.write_bytes(b"second")
            second = _file_fingerprint(str(path))
        self.assertNotEqual(first, second)

    def test_real_stereo_wave_decode(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stereo.wav"
            samples = array("h", [0, 1000, -1000, 2000, -2000, 3000, -3000, 4000])
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes(samples.tobytes())
            waveform, sample_rate = _decode_audio_file(str(path))
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(waveform.shape, (2, 4))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(torch.isfinite(waveform).all())


if __name__ == "__main__":
    unittest.main()
