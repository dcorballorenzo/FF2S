from FF2S_prod.ml_logic.params import LOCAL_REGISTRY_PATH
import matplotlib.pyplot as plt
import glob
import os
from colorama import Fore, Style

from keras import Model, models


def save_model(model: Model = None,
               suffix :str = 'dev') -> None:
    """
    persist trained model, params and metrics
    """

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save training sketches
    #TO IMPORT FROM THE gan.py file

    # save model
    if model is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH,"models", f"model_{suffix}.h5")
        print(f"- model path: {model_path}")
        model.save(model_path)
        print(f'>Saved: {model_path}')

    print("\n✅ data saved locally")

    return None

def save_training_sketches(X_realA, X_realB , X_fakeB,n_samples,suffix,step):

    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i])

    # save plot to file
    output_path = os.path.join(LOCAL_REGISTRY_PATH,"training_sketches",'plot_%s_%06d.png' % (suffix,(step+1)) )
    plt.savefig(output_path)
    plt.close()

def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model

def save_predictions(y_pred,output_path,suffix,pred_number):
    file_path= os.path.join(output_path, "predict_sketches",'prediction_%s_%03d.png' % (suffix,(pred_number+1)))
    y_norm = (y_pred + 1) / 2
    plt.imsave(file_path,y_norm,format="png")

    return None
