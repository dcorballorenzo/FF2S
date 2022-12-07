import os
from FF2S_prod.ml_logic.preproc import clean_namelist

SKETCH_TRAIN=os.environ.get("SKETCH_TRAIN")
PHOTO_TRAIN=os.environ.get("PHOTO_TRAIN")
SAMPLE_SIZE=os.environ.get("SAMPLE_SIZE")

SKETCH_TEST=os.environ.get("SKETCH_TEST")
PHOTO_TEST=os.environ.get("PHOTO_TEST")

SKETCH_FULL_LIST=clean_namelist(os.listdir(SKETCH_TRAIN))
PHOTO_FULL_LIST=clean_namelist(os.listdir(PHOTO_TRAIN))


def get_photo_sketch_dict(photo_list,sketch_list):
    """Takes 2 lists of matching length as input and returns a dictionary with the
    first list as key and the second list as value (in corresponding order"""

    if len(photo_list)==len(sketch_list):
        return {photo_list[i]:sketch_list[i] for i in range(len(photo_list))}
    else :
        print("The length of the sketch list must match the length of the photo list! Please check the data folders.")
        return None


PHOTO_TO_SKETCH_DICT=get_photo_sketch_dict(PHOTO_FULL_LIST,SKETCH_FULL_LIST)

LOCAL_REGISTRY_PATH=os.environ.get("LOCAL_REGISTRY_PATH")

MODEL=os.environ.get("MODEL")

N_EPOCHS=os.environ.get("N_EPOCHS")
