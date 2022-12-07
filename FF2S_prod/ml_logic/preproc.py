import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def same_size(path):
    '''
    Evaluates whether a folder has all images with the same size or not
    1 ------> All of them have the same shape
    0 ------> There's at least one which doesn't have the same shape
    '''

    img_0 = plt.imread(os.path.join(path, sorted(os.listdir(path))[0]))
    for photo in sorted(os.listdir(path)):
        img_aux = plt.imread(os.path.join(path, photo))
        if img_0.shape != img_aux.shape:
            return 0
    return 1


def normalize(image, max = 255):
    '''
    Applies a MinMax scaling to the data
    '''
    return image/max


def photo_sketch_join(photos_path: str, sketches_path:str):
    '''
    Creates a dataframe containing every pair photo-sketch
    Shape: (number of pairs, 2)
    '''

    # We sort the names of the photos and sketches
    photo_list = sorted(os.listdir(photos_path))
    sketch_list = sorted(os.listdir(sketches_path))

    photo_sketch_dict = dict(photo=photo_list, sketch=sketch_list)

    return pd.DataFrame(photo_sketch_dict)

def photo_sketch_load(photos_path: str, sketches_path:str):
    '''
    Reads the np.arrays for the photos and sketches and
    loads it into a DataFrame
    '''
    df = photo_sketch_join(photos_path, sketches_path)

    photo_arrays = []
    sketch_arrays = []

    for photo in df.photo:
        # Para probar voy a meter photo1 pero luego hay que borrarlo
        photo_array = plt.imread(os.path.join(photos_path, photo))
        photo_arrays.append(normalize(photo_array))

    for sketch in df.sketch:
        sketch_array = plt.imread(os.path.join(sketches_path, sketch))
        sketch_arrays.append(normalize(sketch_array))

    dict_aux = dict(photo=photo_arrays, sketch=sketch_arrays)
    return dict_aux


def clean_namelist(raw_namelist):
#clean the .jpg name list and sort it
    clean_name_list = list()
    for x in raw_namelist:
        if ".jpg" in x:
            clean_name_list.append(x)
    clean_name_list.sort()
    return clean_name_list
