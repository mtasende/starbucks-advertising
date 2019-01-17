import os
from pathlib import Path
import pkg_resources

# MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = pkg_resources.resource_filename('src', 'data')
ROOT_DIR = str(Path(MODULE_DIR).parent.parent)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_RAW = os.path.join(DATA_DIR, 'raw')
DATA_INTERIM = os.path.join(DATA_DIR, 'interim')
DATA_EXTERNAL = os.path.join(DATA_DIR, 'external')
DATA_PROCESSED = os.path.join(DATA_DIR, 'processed')


if not os.path.exists(os.path.join(DATA_INTERIM)):
    os.makedirs(DATA_INTERIM)
if not os.path.exists(os.path.join(DATA_EXTERNAL)):
    os.makedirs(DATA_EXTERNAL)
if not os.path.exists(os.path.join(DATA_PROCESSED)):
    os.makedirs(DATA_PROCESSED)
