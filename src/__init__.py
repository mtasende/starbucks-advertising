import os
from pathlib import Path

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = str(Path(MODULE_DIR).parent)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_RAW = os.path.join(DATA_DIR, 'raw')
DATA_INTERIM = os.path.join(DATA_DIR, 'interim')
DATA_EXTERNAL = os.path.join(DATA_DIR, 'external')
DATA_PROCESSED = os.path.join(DATA_DIR, 'processed')
