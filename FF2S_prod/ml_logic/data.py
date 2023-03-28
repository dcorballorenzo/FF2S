from FF2S_prod.ml_logic.preproc import clean_namelist,get_photo_sketch_dict
from FF2S_prod.ml_logic.params import SAMPLE_SIZE,PHOTO_TRAIN,SKETCH_TRAIN
import os
import numpy as np
import matplotlib.pyplot as plt
import random


def get_photo_sample(photo_path=PHOTO_TRAIN, n_samples = SAMPLE_SIZE ):

    photo_list=clean_namelist(os.listdir(photo_path))
    photo_sample = random.sample(photo_list,int(n_samples))
    return photo_sample

def get_sketch_sample (photo_sample,photo_path=PHOTO_TRAIN,sketch_path=SKETCH_TRAIN ):
    translation_dict=get_photo_sketch_dict(photo_path,sketch_path)
    sketch_sample = [translation_dict[photo] for photo in photo_sample]
    return sketch_sample


def load_images(image_list, path):
    """Takes as input a list of images names and returns an array containing the images
    """
    data_list = []
    for filename in image_list:
        data_list.append(np.asarray(plt.imread(os.path.join(path, filename))))

    return np.asarray(data_list)

def load_array():
   photo_sample = get_photo_sample()
   sketch_sample=get_sketch_sample(photo_sample)
   photo_array = load_images(photo_sample, PHOTO_TRAIN)
   photo_array=(photo_array-127.5)/127.5

   sketch_array = load_images(sketch_sample, SKETCH_TRAIN)
   sketch_array=(sketch_array-127.5)/127.5

   return photo_array, sketch_array
