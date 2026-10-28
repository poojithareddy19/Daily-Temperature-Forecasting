import hashlib

import pandas as pd
import pytest

from src.data import ingest

SOURCE = b'"Date","Temp"\r\n"1981-01-01",20.7\r\n"1981-01-02",17.9\r\n'


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture
def raw_path(tmp_path, monkeypatch):
    """Point ingest at a temp dir and pin the checksum to SOURCE; no network."""
    cfg = {
        "data": {
            "raw_url": "https://example.invalid/data.csv",
            "raw_sha256": hashlib.sha256(SOURCE).hexdigest(),
            "raw_path": "raw/data.csv",
        }
    }
    monkeypatch.setattr(ingest, "load_config", lambda: cfg)
    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    return tmp_path / "raw" / "data.csv"


def test_matching_checksum_writes_file(raw_path, monkeypatch):
    monkeypatch.setattr(ingest.requests, "get", lambda url, timeout: FakeResponse(SOURCE))

    ingest.ingest()

    df = pd.read_csv(raw_path)
    assert list(df.columns) == ["date", "temp"]
    assert df["temp"].tolist() == [20.7, 17.9]
    assert b"\r\n" not in raw_path.read_bytes()


def test_mismatching_checksum_raises_and_writes_nothing(raw_path, monkeypatch):
    tampered = SOURCE.replace(b"20.7", b"99.9")
    monkeypatch.setattr(ingest.requests, "get", lambda url, timeout: FakeResponse(tampered))

    with pytest.raises(ingest.ChecksumMismatchError, match="sha256"):
        ingest.ingest()

    assert not raw_path.exists()
    assert not raw_path.parent.exists()
