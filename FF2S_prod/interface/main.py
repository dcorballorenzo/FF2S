import numpy as np
import pandas as pd

from colorama import Fore, Style

from FF2S_prod.ml_logic.gan import train_model
from FF2S_prod.ml_logic.registry import load_model, save_model
from FF2S_prod.ml_logic.data import get_photo_sample,get_sketch_sample, load_images
from FF2S_prod.ml_logic.gan import define_discriminator, define_generator, define_gan, train_model
from FF2S_prod.ml_logic.params import PHOTO_PATH, SKETCH_PATH
import matplotlib.pyplot as plt


def preprocess():
   photo_sample = get_photo_sample()
   sketch_sample=get_sketch_sample(photo_sample)
   photo_array = load_images(photo_sample, PHOTO_PATH)
   photo_array=(photo_array-127.5)/127.5

   sketch_array = load_images(sketch_sample, SKETCH_PATH)
   sketch_array=(sketch_array-127.5)/127.5

   return photo_array, sketch_array


def train():
    d_model = define_discriminator()
    g_model = define_generator()
    gan_model = define_gan(g_model, d_model)
    dataset = preprocess()
    train_model(d_model, g_model, gan_model, dataset)



def pred(visualize=False):
    """
    Make a prediction using the latest trained model
    """

    model = load_model('/Users/alicepannequin/code/dcorballorenzo/FF2S/training_outputs/models/model_20221205-165308.h5')
    X_new_photo = get_photo_sample(n_samples=1)
    X_pred = load_images(X_new_photo,PHOTO_PATH)

    X_processed = (X_pred - 127.5)/127.5
    if len(X_processed.shape)==3: X_processed = X_processed.reshape((1, 256, 256, 3))
    # X_processed = X_processed[0]

    y_pred = model.predict(X_processed)[0]

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    if visualize:
        plt.imshow(y_pred)
        plt.show()
    else:
        return y_pred



if __name__=="__main__":
    preprocess()
    train()
    pred(visualize=True)
