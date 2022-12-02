from FF2S_prod.ml_logic.preproc import clean_namelist
from FF2S_prod.ml_logic.params import SAMPLE_SIZE
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def load_images(path, n_samples = SAMPLE_SIZE):
    """Takes as input a path to images and returns
    """

    #get the list of filenames
    raw_name_list=os.listdir(path)

    #clean the filenames list
    clean_name_list = clean_namelist(raw_name_list)

    #get a sample list of photos

    sample_list = random.sample(clean_name_list,n_samples)

	# enumerate filenames in directory

    data_list = [np.asarray(plt.imread(os.path.join(path,filename))) for filename in sample_list]

    return np.asarray(data_list)
