from pathlib import Path


BASE_DIR: Path = Path(__name__).parent.resolve()
DATA_DIR: Path = BASE_DIR / 'data'

TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH: Path = DATA_DIR / 'tqbr_top_listlevel_securities.json'
HISTORY_JSON_PATH: Path = DATA_DIR / 'history.json'
MSFO_DATA_JSON_PATH: Path = DATA_DIR / 'msfo_data.json'
