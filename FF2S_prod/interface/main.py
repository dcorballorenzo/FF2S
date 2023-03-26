import numpy as np
import pandas as pd
import shutil

from colorama import Fore, Style

from FF2S_prod.ml_logic.gan import train_model
from FF2S_prod.ml_logic.preproc import clean_namelist,resize,get_photo_sketch_dict
from FF2S_prod.ml_logic.registry import load_model, save_model, save_predictions
from FF2S_prod.ml_logic.data import get_photo_sample,get_sketch_sample, load_images, clean_namelist,load_array
from FF2S_prod.ml_logic.cycle_gan import define_discriminator_cycle, define_generator_cycle, define_composite_model, train_model_cycle
from FF2S_prod.ml_logic.gan import define_discriminator, define_generator, define_gan, train_model
from FF2S_prod.ml_logic.params import PHOTO_RAW, SKETCH_RAW,PHOTO_TRAIN, SKETCH_TRAIN, PHOTO_TEST, SKETCH_TEST, MODEL,LOCAL_REGISTRY_PATH,N_PREDICT,TTS_RATE

import matplotlib.pyplot as plt
import sys
import os


def preprocess():

    print("🤖 Starting preprocessing ...")
    #Initialize two count variables that will be useful for the loops
    image_cpt=0

    #Clean list of folder names
    photo_folder_list=os.listdir(PHOTO_RAW)
    sketch_folder_list=os.listdir(SKETCH_RAW)

    if '.DS_Store' in photo_folder_list:
        photo_folder_list.remove('.DS_Store')
    photo_folder_list.sort()

    if '.DS_Store' in sketch_folder_list:
        sketch_folder_list.remove('.DS_Store')
    sketch_folder_list.sort()

    for photo_folder,sketch_folder in zip(photo_folder_list,sketch_folder_list):

    #Define the paths to raw photos and raw sketches
        photo_path=os.path.join(PHOTO_RAW,photo_folder)
        sketch_path=os.path.join(SKETCH_RAW,sketch_folder)

    #List of image names
        photo_list=clean_namelist(os.listdir(photo_path))
        sketch_list=clean_namelist(os.listdir(sketch_path))


        if len(photo_list)!=len(sketch_list):
            print("⚠️ The input folders don't have the same number of images!")
            return None

    # Resize the images and save them temporarily
        resize(source_path=photo_path,destination_path=PHOTO_RAW,name_list=photo_list,output_name="photo",cpt=image_cpt)
        resize(source_path=sketch_path,destination_path=SKETCH_RAW,name_list=sketch_list,output_name="sketch",cpt=image_cpt)
        image_cpt+=len(photo_list)

    print("\n✅ Preprocessing done: ")

def train_test_split(rate=TTS_RATE):

    print("🖖 Starting Train-Test-Split ...")
    # Get the clean lists for photos and sketches

    photo_list_clean=clean_namelist(os.listdir(PHOTO_RAW))
    sketch_list_clean=clean_namelist(os.listdir(SKETCH_RAW))

    #create a translation dictionary from the two clean lists
    translation_dict=get_photo_sketch_dict(photo_list_clean,sketch_list_clean)

    # sample the test photos
    photo_test=get_photo_sample(photo_list_clean,round(2006*rate))
    photo_test.sort()
    #get the sample of corresponding sketches
    sketch_test=get_sketch_sample(photo_test,translation_dict)
    sketch_test.sort()

    #get the train photos and sketches by difference

    photo_train=[photo for photo in photo_list_clean if photo not in photo_test]
    photo_train.sort()

    sketch_train=get_sketch_sample(photo_train,translation_dict)
    sketch_train.sort()

    print("📸 Sampling done")

    #Check that all the samples have the same size
    if len(photo_train)!=len(sketch_train):
            print("⚠️ The training folders don't have the same number of images!")
            return None

    if len(photo_test)!=len(sketch_test):
            print("⚠️ The testing folders don't have the same number of images!")
            return None
    print("🚀 Moving the images to Train/Test folders ...")
    #Move the photos to the train folders
    for p_train,s_train in zip(photo_train,sketch_train):
        shutil.move(os.path.join(PHOTO_RAW,p_train),os.path.join(PHOTO_TRAIN,p_train))
        shutil.move(os.path.join(SKETCH_RAW,s_train),os.path.join(SKETCH_TRAIN,s_train))

    #Move the images to the test folders

    for p_test,s_test in zip(photo_test,sketch_test):
        shutil.move(os.path.join(PHOTO_RAW,p_test),os.path.join(PHOTO_TEST,p_test))
        shutil.move(os.path.join(SKETCH_RAW,s_test),os.path.join(SKETCH_TEST,s_test))

    print("\n✅ Train-Test-Split done: ")

def train(suffix='dev'):
    if MODEL=="CGAN":
        d_model = define_discriminator()
        g_model = define_generator()
        gan_model = define_gan(g_model, d_model)
        dataset = load_array()
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
    train_test_split()
    train(suffix=sys.argv[1])
    pred(model_suffix=sys.argv[1])
