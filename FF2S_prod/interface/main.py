import numpy as np
import pandas as pd

from colorama import Fore, Style

from FF2S_prod.ml_logic.gan import train_model
from FF2S_prod.ml_logic.registry import load_model, save_model
from FF2S_prod.ml_logic.data import get_photo_sample,get_sketch_sample, load_images
from FF2S_prod.ml_logic.cycle_gan import define_discriminator_cycle, define_generator_cycle, define_composite_model, train_model_cycle
from FF2S_prod.ml_logic.gan import define_discriminator, define_generator, define_gan, train_model
from FF2S_prod.ml_logic.params import PHOTO_PATH, SKETCH_PATH,MODEL,LOCAL_REGISTRY_PATH
import matplotlib.pyplot as plt
import sys
import os


def preprocess():
   photo_sample = get_photo_sample()
   sketch_sample=get_sketch_sample(photo_sample)
   photo_array = load_images(photo_sample, PHOTO_PATH)
   photo_array=(photo_array-127.5)/127.5

   sketch_array = load_images(sketch_sample, SKETCH_PATH)
   sketch_array=(sketch_array-127.5)/127.5

   return photo_array, sketch_array


def train():
    if MODEL=="CGAN":
        d_model = define_discriminator()
        g_model = define_generator()
        gan_model = define_gan(g_model, d_model)
        dataset = preprocess()
        model = train_model(d_model, g_model, gan_model, dataset)
        return model

    elif MODEL =="CycleGAN":
        # generator: B -> A
        g_model_AtoB = define_generator_cycle()
        # generator: B -> A
        g_model_BtoA = define_generator_cycle()
        # discriminator: A -> [real/fake]
        d_model_A = define_discriminator_cycle()
        # discriminator: B -> [real/fake]
        d_model_B = define_discriminator_cycle()
        # composite: A -> B -> [real/fake, A]
        c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA)
        # composite: B -> A -> [real/fake, B]
        c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB)
        dataset = preprocess()
        train_model_cycle(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)



def pred(visualize=False):
    """
    Make a prediction using the latest trained model
    """
    model = load_model(os.path.join("/Users/alicepannequin/code/dcorballorenzo/FF2S/training_outputs/models/model_20221205-174911.h5"))
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
