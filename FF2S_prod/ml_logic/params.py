import os

#Path to raw data

PHOTO_RAW=os.path.join(os.getcwd(),"data","raw_data","photo")
SKETCH_RAW=os.path.join(os.getcwd(),"data","raw_data","sketch")

#Path to preprocessed data

SKETCH_TRAIN=os.path.join(os.getcwd(),"data","preproc_data","train","sketch")
PHOTO_TRAIN=os.path.join(os.getcwd(),"data","preproc_data","train","photo")

SKETCH_TEST=os.path.join(os.getcwd(),"data","preproc_data","test","sketch")
PHOTO_TEST=os.path.join(os.getcwd(),"data","preproc_data","test","photo")

#Ratio used for the Train-Test-Split

TTS_RATIO=0.2

#Type of model to use ("CGAN" or "CycleGAN")

MODEL="CGAN"

#Path to save outputs

LOCAL_REGISTRY_PATH=os.path.join(os.getcwd(),"training_outputs")


#parameters for the model training (name of the model, number of images and number of epochs)
SAMPLE_SIZE=10
N_EPOCHS=5
PREDICT_NAME=f"{MODEL}_{SAMPLE_SIZE}i_{N_EPOCHS}e"


#Number of images to predict with the model
N_PREDICT=2
