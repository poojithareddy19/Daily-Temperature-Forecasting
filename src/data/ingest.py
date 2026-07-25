import pandas as pd
from src.config import PROJECT_ROOT, load_config
from src.logger import get_logger
logger = get_logger(__name__)
def ingest() -> pd.DataFrame:
    cfg = load_config()
    url = cfg["data"]["raw_url"]
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", url)
    df = pd.read_csv(url)
    df.columns = ["date", "temp"]
    df.to_csv(raw_path, index=False)
    logger.info("Saved %d rows to %s", len(df), raw_path)
    return df
if __name__ == "__main__":
    ingest()