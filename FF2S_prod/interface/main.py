import numpy as np
import pandas as pd

from colorama import Fore, Style

from FF2S_prod.ml_logic.gan import train_model
from FF2S_prod.ml_logic.registry import load_model, save_model, save_predictions
from FF2S_prod.ml_logic.data import get_photo_sample,get_sketch_sample, load_images, clean_namelist
from FF2S_prod.ml_logic.cycle_gan import define_discriminator_cycle, define_generator_cycle, define_composite_model, train_model_cycle
from FF2S_prod.ml_logic.gan import define_discriminator, define_generator, define_gan, train_model
from FF2S_prod.ml_logic.params import PHOTO_TRAIN, SKETCH_TRAIN, PHOTO_TEST, SKETCH_TEST, MODEL,LOCAL_REGISTRY_PATH,N_PREDICT

import matplotlib.pyplot as plt
import sys
import os


def preprocess():
   photo_sample = get_photo_sample()
   sketch_sample=get_sketch_sample(photo_sample)
   photo_array = load_images(photo_sample, PHOTO_TRAIN)
   photo_array=(photo_array-127.5)/127.5

   sketch_array = load_images(sketch_sample, SKETCH_TRAIN)
   sketch_array=(sketch_array-127.5)/127.5

   return photo_array, sketch_array


def train(suffix='dev'):
    if MODEL=="CGAN":
        d_model = define_discriminator()
        g_model = define_generator()
        gan_model = define_gan(g_model, d_model)
        dataset = preprocess()
        model = train_model(d_model, g_model, gan_model, dataset,suffix=suffix)
        return model

# train_model(d_model, g_model, gan_model, dataset, n_epochs=N_EPOCHS, n_batch=1,suffix='dev'):

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
        train_model_cycle(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset,suffix=suffix)



def pred(model_suffix='dev',n_predict=N_PREDICT):
    """
    Make a prediction using the latest trained model
    """
    model = load_model(os.path.join(LOCAL_REGISTRY_PATH,"models",f"model_{model_suffix}.h5"))
    X_new_photo = get_photo_sample(photo_list=clean_namelist(os.listdir(PHOTO_TEST)),n_samples=n_predict)

    X_pred = load_images(X_new_photo,PHOTO_TEST)

    X_processed = (X_pred - 127.5)/127.5
    if len(X_processed.shape)==3: X_processed = X_processed.reshape((1, 256, 256, 3))
    # X_processed = X_processed[0]

    for i in range(int(n_predict)):
        y_pred = model.predict(X_processed)[i]
        output_path= os.path.join(LOCAL_REGISTRY_PATH, "predict_sketches",'prediction_%03d_%s.png' % ((i+1), model_suffix))
        save_predictions(y_pred=y_pred,output_path=output_path)

    print("\n✅ prediction done: ")


if __name__=="__main__":
    preprocess()
    train(suffix=sys.argv[1])
    pred(model_suffix=sys.argv[1])
