from FF2S_prod.ml_logic.preproc import clean_namelist
from FF2S_prod.ml_logic.params import SAMPLE_SIZE, PHOTO_TO_SKETCH_DICT,PHOTO_FULL_LIST
import os
import numpy as np
import matplotlib.pyplot as plt
import random


def get_photo_sample(photo_list=PHOTO_FULL_LIST, n_samples = SAMPLE_SIZE ):

    photo_sample = random.sample(photo_list,int(n_samples))
    return photo_sample

def get_sketch_sample (photo_sample,translation_dict= PHOTO_TO_SKETCH_DICT ):

    sketch_sample = [translation_dict[photo] for photo in photo_sample]
    return sketch_sample


def load_images(image_list, path):
    """Takes as input a list of images names and returns an array containing the images
    """
    data_list = []
    for filename in image_list:
        data_list.append(np.asarray(plt.imread(os.path.join(path, filename))))

    return np.asarray(data_list)
