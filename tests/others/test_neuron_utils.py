# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


# Patch torch-neuronx version at import time so the module loads without the SDK installed.
with patch("importlib.metadata.version", return_value="2.9.0"):
    from diffusers.utils.neuron_utils import NeuronCache, _neuronx_sdk_version

_ENV_VAR = "TORCH_NEURONX_NEFF_CACHE_DIR"


class TestNeuronxSdkVersion(unittest.TestCase):
    def test_returns_major_minor(self):
        with patch("importlib.metadata.version", return_value="2.9.0.1"):
            assert _neuronx_sdk_version() == "2.9"

    def test_handles_two_part_version(self):
        with patch("importlib.metadata.version", return_value="3.0"):
            assert _neuronx_sdk_version() == "3.0"


class TestNeuronCacheSubfolderBuilding(unittest.TestCase):
    def _make(self, **kwargs):
        with patch("importlib.metadata.version", return_value="2.9.0"):
            return NeuronCache("owner/repo", **kwargs)

    def test_explicit_subfolder(self):
        c = self._make(subfolder="custom/path")
        assert c.subfolder == "custom/path"

    def test_auto_subfolder_with_mode_and_resolution(self):
        c = self._make(mode="eager", height=512, width=512)
        assert c.subfolder == "sdk2.9/eager_512x512"

    def test_auto_subfolder_compile_mode(self):
        c = self._make(mode="compile", height=256, width=256)
        assert c.subfolder == "sdk2.9/compile_256x256"

    def test_auto_subfolder_falls_back_to_sdk_only(self):
        c = self._make()
        assert c.subfolder == "sdk2.9"

    def test_auto_subfolder_partial_args_falls_back(self):
        # Only mode, no height/width → fall back to sdk-only
        c = self._make(mode="eager")
        assert c.subfolder == "sdk2.9"

    def test_explicit_local_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            c = self._make(local_cache_dir=tmpdir)
            assert c.local_cache_dir == Path(tmpdir)

    def test_default_local_cache_dir(self):
        c = self._make(mode="eager", height=512, width=512)
        expected = Path.home() / ".cache" / "diffusers" / "neuron" / "owner--repo" / "sdk2.9" / "eager_512x512"
        assert c.local_cache_dir == expected


class TestNeuronCacheContextManager(unittest.TestCase):
    """Tests __enter__ / __exit__ without touching the real HF Hub."""

    def _make(self, local_cache_dir, **kwargs):
        with patch("importlib.metadata.version", return_value="2.9.0"):
            return NeuronCache("owner/repo", local_cache_dir=local_cache_dir, **kwargs)

    def _fake_fs(self, remote_files=None):
        """Return a mock HfFileSystem with configurable remote file list."""
        fs = MagicMock()
        if remote_files is None:
            remote_files = ["owner/repo/sdk2.9/a.neff", "owner/repo/sdk2.9/b.neff"]
        fs.ls.return_value = [{"name": f, "type": "file"} for f in remote_files]
        fs.get = MagicMock()
        return fs

    # ------------------------------------------------------------------
    # env-var management
    # ------------------------------------------------------------------

    def test_env_var_set_on_enter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            prev = os.environ.pop(_ENV_VAR, None)
            try:
                with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=self._fake_fs([])):
                    with cache:
                        assert os.environ[_ENV_VAR] == tmpdir
            finally:
                if prev is None:
                    os.environ.pop(_ENV_VAR, None)
                else:
                    os.environ[_ENV_VAR] = prev

    def test_env_var_restored_on_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            os.environ[_ENV_VAR] = "/previous/path"
            try:
                with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=self._fake_fs([])):
                    with cache:
                        pass
                assert os.environ[_ENV_VAR] == "/previous/path"
            finally:
                os.environ.pop(_ENV_VAR, None)

    def test_env_var_removed_when_not_set_before(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            os.environ.pop(_ENV_VAR, None)
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=self._fake_fs([])):
                with cache:
                    assert _ENV_VAR in os.environ
            assert _ENV_VAR not in os.environ

    # ------------------------------------------------------------------
    # lazy download logic
    # ------------------------------------------------------------------

    def test_downloads_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = self._fake_fs(["owner/repo/sdk2.9/a.neff", "owner/repo/sdk2.9/b.neff"])
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:
                    pass
            assert fake_fs.get.call_count == 2

    def test_skips_already_cached_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create one file so it looks already cached.
            Path(tmpdir, "a.neff").touch()
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = self._fake_fs(["owner/repo/sdk2.9/a.neff", "owner/repo/sdk2.9/b.neff"])
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:
                    pass
            # Only b.neff should have been fetched.
            assert fake_fs.get.call_count == 1
            call_args = fake_fs.get.call_args[0]
            assert "b.neff" in call_args[0]

    def test_skips_all_when_all_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.neff").touch()
            Path(tmpdir, "b.neff").touch()
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = self._fake_fs(["owner/repo/sdk2.9/a.neff", "owner/repo/sdk2.9/b.neff"])
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:
                    pass
            fake_fs.get.assert_not_called()

    def test_no_download_when_remote_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = self._fake_fs([])
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:
                    pass
            fake_fs.get.assert_not_called()

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    def test_repo_not_found_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = MagicMock()
            fake_fs.ls.side_effect = RepositoryNotFoundError("owner/repo", response=MagicMock())
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:  # should not raise
                    pass

    def test_entry_not_found_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = MagicMock()
            fake_fs.ls.side_effect = EntryNotFoundError("owner/repo/sdk2.9")
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:  # should not raise
                    pass

    def test_file_not_found_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            fake_fs = MagicMock()
            fake_fs.ls.side_effect = FileNotFoundError
            with patch("diffusers.utils.neuron_utils.HfFileSystem", return_value=fake_fs):
                with cache:  # should not raise
                    pass


class TestNeuronCachePushToHub(unittest.TestCase):
    def _make(self, local_cache_dir, **kwargs):
        with patch("importlib.metadata.version", return_value="2.9.0"):
            return NeuronCache("owner/repo", local_cache_dir=local_cache_dir, **kwargs)

    def test_push_uploads_neff_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "x.neff").touch()
            Path(tmpdir, "y.neff").touch()
            cache = self._make(local_cache_dir=tmpdir, mode="eager", height=512, width=512)
            with patch("diffusers.utils.neuron_utils.upload_folder", return_value="https://hub.url") as mock_upload:
                url = cache.push_to_hub()
            assert url == "https://hub.url"
            mock_upload.assert_called_once()
            call_kwargs = mock_upload.call_args[1]
            assert call_kwargs["repo_id"] == "owner/repo"
            assert call_kwargs["path_in_repo"] == "sdk2.9/eager_512x512"
            assert call_kwargs["allow_patterns"] == ["*.neff"]

    def test_push_with_explicit_subfolder_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "x.neff").touch()
            cache = self._make(local_cache_dir=tmpdir)
            with patch("diffusers.utils.neuron_utils.upload_folder", return_value="https://hub.url") as mock_upload:
                cache.push_to_hub(subfolder="custom/override")
            call_kwargs = mock_upload.call_args[1]
            assert call_kwargs["path_in_repo"] == "custom/override"

    def test_push_returns_empty_when_no_neff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self._make(local_cache_dir=tmpdir)
            with patch("diffusers.utils.neuron_utils.upload_folder") as mock_upload:
                url = cache.push_to_hub()
            assert url == ""
            mock_upload.assert_not_called()

    def test_push_custom_commit_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "x.neff").touch()
            cache = self._make(local_cache_dir=tmpdir)
            with patch("diffusers.utils.neuron_utils.upload_folder", return_value="https://hub.url") as mock_upload:
                cache.push_to_hub(commit_message="My custom commit")
            call_kwargs = mock_upload.call_args[1]
            assert call_kwargs["commit_message"] == "My custom commit"


if __name__ == "__main__":
    unittest.main()
