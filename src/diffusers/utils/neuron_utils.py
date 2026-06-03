import importlib.metadata
import os
from pathlib import Path

from huggingface_hub import HfFileSystem, upload_folder
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from .logging import get_logger


logger = get_logger(__name__)

_ENV_VAR = "TORCH_NEURONX_NEFF_CACHE_DIR"


def _neuronx_sdk_version() -> str:
    """Return 'major.minor' of the installed torch-neuronx, e.g. '2.9'."""
    raw = importlib.metadata.version("torch-neuronx")
    parts = raw.split(".")
    return f"{parts[0]}.{parts[1]}"


class NeuronCache:
    """Context manager that backs ``TORCH_NEURONX_NEFF_CACHE_DIR`` with a HF Hub repo.

    On enter, NEFF files in the requested subfolder of *cache_repo_id* are
    downloaded lazily to *local_cache_dir* — only files that are not already
    present on disk are fetched. ``TORCH_NEURONX_NEFF_CACHE_DIR`` is then
    pointed at that directory so the Neuron SDK can load pre-compiled artifacts
    and skip neuronx-cc recompilation entirely. On exit the original env-var
    value is restored.

    The subfolder encodes the Neuron SDK version, compilation mode, and image
    resolution so that incompatible NEFFs are never mixed::

        {cache_repo_id}/
        ├── sdk2.9/
        │   ├── eager_512x512/   ← eager mode, 512×512
        │   └── compile_256x256/ ← torch.compile mode, 256×256
        └── sdk2.10/
            └── ...

    When *subfolder* is ``None``, it is built automatically as
    ``sdk{major}.{minor}/{mode}_{height}x{width}``. Pass *subfolder* explicitly
    to override.

    Args:
        cache_repo_id: HF Hub repo ID that stores pre-compiled NEFF files
            (e.g. ``"aws-neuron/flux2-klein-neff-cache"``).
        subfolder: Path within *cache_repo_id* that contains the NEFF files for
            this run. Auto-built from *mode* / *height* / *width* when ``None``.
        mode: ``"eager"`` or ``"compile"``. Used for auto subfolder construction.
        height: Image height in pixels. Used for auto subfolder construction.
        width: Image width in pixels. Used for auto subfolder construction.
        local_cache_dir: Local directory that receives the downloaded NEFF files
            and is set as ``TORCH_NEURONX_NEFF_CACHE_DIR``. Defaults to
            ``~/.cache/diffusers/neuron/{cache_repo_id}/{subfolder}``.
        token: HF Hub authentication token (required for private repos).

    Example::

        with NeuronCache(
            "aws-neuron/flux2-klein-neff-cache",
            mode="compile",
            height=256,
            width=256,
        ) as cache:
            pipe = load_pipeline("compile")
            pipe(prompt, height=256, width=256)
            # Upload newly compiled NEFFs for others to reuse:
            cache.push_to_hub()
    """

    def __init__(
        self,
        cache_repo_id: str,
        subfolder: str | None = None,
        mode: str | None = None,
        height: int | None = None,
        width: int | None = None,
        local_cache_dir: str | None = None,
        token: str | None = None,
    ):
        self.cache_repo_id = cache_repo_id
        self.token = token

        sdk_ver = _neuronx_sdk_version()
        if subfolder is not None:
            self.subfolder = subfolder
        elif mode is not None and height is not None and width is not None:
            self.subfolder = f"sdk{sdk_ver}/{mode}_{height}x{width}"
        else:
            self.subfolder = f"sdk{sdk_ver}"

        if local_cache_dir is not None:
            self.local_cache_dir = Path(local_cache_dir)
        else:
            safe_repo = cache_repo_id.replace("/", "--")
            self.local_cache_dir = Path.home() / ".cache" / "diffusers" / "neuron" / safe_repo / self.subfolder

        self._prev_env: str | None = None

    def __enter__(self) -> "NeuronCache":
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy download: fetch only files not already present on disk.
        remote_path = f"{self.cache_repo_id}/{self.subfolder}"
        try:
            fs = HfFileSystem(token=self.token)
            entries = fs.ls(remote_path, detail=True)
            remote_files = [e["name"] for e in entries if e["type"] == "file"]
            if not remote_files:
                logger.warning(f"No NEFF files found in '{remote_path}' — will compile from scratch.")
            else:
                downloaded = 0
                for remote_file in remote_files:
                    filename = Path(remote_file).name
                    local_path = self.local_cache_dir / filename
                    if not local_path.exists():
                        logger.info(f"Downloading NEFF: {filename}")
                        fs.get(remote_file, str(local_path))
                        downloaded += 1
                skipped = len(remote_files) - downloaded
                logger.info(
                    f"NeuronCache: {downloaded} NEFF(s) downloaded, {skipped} already cached "
                    f"(total {len(remote_files)}) from '{remote_path}'."
                )
        except (RepositoryNotFoundError, EntryNotFoundError, FileNotFoundError) as e:
            logger.warning(
                f"NeuronCache: could not fetch NEFFs from '{remote_path}' ({e}). Will compile from scratch."
            )

        # Point the SDK at the local dir.
        self._prev_env = os.environ.get(_ENV_VAR)
        os.environ[_ENV_VAR] = str(self.local_cache_dir)
        logger.info(f"NeuronCache: {_ENV_VAR}={self.local_cache_dir}")
        return self

    def __exit__(self, *args):
        if self._prev_env is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = self._prev_env

    def push_to_hub(
        self,
        subfolder: str | None = None,
        commit_message: str = "Update NEFF cache",
        private: bool = False,
    ) -> str:
        """Upload all NEFF files from the local cache to *cache_repo_id* on the Hub.

        Returns the Hub URL of the resulting commit.
        """
        target = subfolder or self.subfolder
        neff_files = list(self.local_cache_dir.glob("*.neff"))
        if not neff_files:
            logger.warning(
                f"NeuronCache.push_to_hub: no *.neff files found in '{self.local_cache_dir}' — nothing uploaded."
            )
            return ""

        logger.info(f"NeuronCache: uploading {len(neff_files)} NEFF(s) to '{self.cache_repo_id}/{target}' ...")
        url = upload_folder(
            repo_id=self.cache_repo_id,
            folder_path=str(self.local_cache_dir),
            path_in_repo=target,
            allow_patterns=["*.neff"],
            token=self.token,
            commit_message=commit_message,
            create_pr=False,
        )
        logger.info(f"NeuronCache: upload complete → {url}")
        return url
