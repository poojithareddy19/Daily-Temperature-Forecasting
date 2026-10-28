import hashlib
import io

import pandas as pd
import requests

from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger

logger = get_logger(__name__)

TIMEOUT = 60  # seconds


class ChecksumMismatchError(ValueError):
    """The downloaded file is not the one the pipeline was built from."""


def ingest() -> pd.DataFrame:
    cfg = load_config()
    url = cfg["data"]["raw_url"]
    expected = cfg["data"]["raw_sha256"]
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    content = response.content

    # Checked before anything is written, so a changed upstream file never reaches the pipeline.
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ChecksumMismatchError(
            f"sha256 of {url} is {actual}, expected {expected} (data.raw_sha256 in "
            "configs/config.yaml). The upstream file changed; check it before updating the hash."
        )

    df = pd.read_csv(io.BytesIO(content))
    df.columns = ["date", "temp"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed line ending, so the file (and its DVC hash) is the same on every OS.
    df.to_csv(raw_path, index=False, lineterminator="\n")
    logger.info("Saved %d rows to %s", len(df), raw_path)
    return df


if __name__ == "__main__":
    ingest()
