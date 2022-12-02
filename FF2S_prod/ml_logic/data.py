from FF2S_prod.ml_logic.preproc import clean_namelist
from FF2S_prod.ml_logic.params import SAMPLE_SIZE, PHOTO_TO_SKETCH_DICT,PHOTO_FULL_LIST,SKETCH_FULL_LIST
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def get_photo_sketch_dict(photo_list,sketch_list):
    """Takes 2 lists of matching length as input and returns a dictionary with the
    first list as key and the second list as value (in corresponding order"""

    if len(photo_list)==len(sketch_list):
        return {photo_list[i]:sketch_list[i] for i in range(len(photo_list))}
    else :
        print("The length of the sketch list must match the length of the photo list! Please check the data folders.")
        return None

def get_sample(sketch_list=SKETCH_FULL_LIST,photo_list=PHOTO_FULL_LIST,translation_dict= PHOTO_TO_SKETCH_DICT, n_samples = SAMPLE_SIZE ):
    """ Returns """

    photo_sample = random.sample(photo_list,n_samples)
    sketch_sample = [translation_dict[photo] for photo in photo_sample]

    return photo_sample,sketch_sample



def load_images(image_list):
    """Takes as input a list of images names and returns an array containing the images
    """

    data_list = [np.asarray(plt.imread(filename)) for filename in image_list]

    return np.asarray(data_list)
