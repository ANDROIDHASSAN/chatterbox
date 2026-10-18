"""prepare_conditionals() accepts a URL reference wav by downloading it to a
temp file first, but only from an allowed host, and always cleans the temp
file up. _local_wav_path() holds that logic; this execs it in isolation (like
test_version_fallback.py) instead of importing chatterbox.mtl_tts, which pulls
in torch/librosa/perth and the rest of the model stack."""

import os
import unittest
from unittest import mock

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "chatterbox", "mtl_tts.py")


def _load_local_wav_path():
    """Exec just the imports and _local_wav_path() out of mtl_tts.py, skipping
    everything below it (the torch/librosa-dependent model code)."""
    with open(SRC, "r", encoding="utf-8") as f:
        source = f.read()

    marker = "REPO_ID = "
    cutoff = source.index(marker)

    namespace = {"__name__": "chatterbox._local_wav_path_test_module"}
    exec(source[:cutoff], namespace)  # noqa: S102 - trusted local source file
    return namespace["_local_wav_path"], namespace["ALLOWED_REFERENCE_WAV_HOSTS"]


_local_wav_path, ALLOWED_REFERENCE_WAV_HOSTS = _load_local_wav_path()


class LocalWavPathTests(unittest.TestCase):
    def test_local_path_is_used_unchanged(self):
        with _local_wav_path("/tmp/some_voice.wav") as path:
            self.assertEqual(path, "/tmp/some_voice.wav")

    def test_allowed_host_url_is_downloaded_and_cleaned_up(self):
        host = next(iter(ALLOWED_REFERENCE_WAV_HOSTS))
        url = f"https://{host}/chatterbox-demo-samples/mtl_prompts/en_f1.flac"

        with mock.patch("urllib.request.urlretrieve") as urlretrieve:
            with _local_wav_path(url) as path:
                self.assertTrue(os.path.exists(path))
                self.assertNotEqual(path, url)
                urlretrieve.assert_called_once_with(url, path)

        # The temp file is removed once the context exits.
        self.assertFalse(os.path.exists(path))

    def test_disallowed_host_is_rejected_without_downloading(self):
        url = "https://attacker.example.com/payload.wav"

        with mock.patch("urllib.request.urlretrieve") as urlretrieve:
            with self.assertRaisesRegex(ValueError, "Disallowed hostname"):
                with _local_wav_path(url):
                    pass

        urlretrieve.assert_not_called()

    def test_url_with_no_hostname_is_rejected_cleanly(self):
        # e.g. "https:///x.wav" - urlparse().hostname is None, not a string.
        with self.assertRaisesRegex(ValueError, "Disallowed hostname"):
            with _local_wav_path("https:///x.wav"):
                pass


if __name__ == "__main__":
    unittest.main()
