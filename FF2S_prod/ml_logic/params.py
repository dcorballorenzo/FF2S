import os
from FF2S_prod.ml_logic.preproc import clean_namelist
from FF2S_prod.ml_logic.data import get_photo_sketch_dict

SKETCH_PATH= os.environ.get("SKETCH_PATH")
PHOTO_PATH= os.environ.get("PHOTO_PATH")
SAMPLE_SIZE= os.environ.get("SAMPLE_SIZE")

SKETCH_FULL_LIST=clean_namelist(os.listdir(SKETCH_PATH))
PHOTO_FULL_LIST=clean_namelist(os.listdir(PHOTO_PATH))

PHOTO_TO_SKETCH_DICT= get_photo_sketch_dict(PHOTO_FULL_LIST,SKETCH_FULL_LIST)
