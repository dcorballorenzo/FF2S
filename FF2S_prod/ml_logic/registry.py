from FF2S_prod.ml_logic.params import LOCAL_REGISTRY_PATH


import glob
import os
import time
import pickle

from colorama import Fore, Style


from tensorflow.keras import Model, models


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None,
               suffix :str = 'dev') -> None:
    """
    persist trained model, params and metrics
    """

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", suffix + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", suffix + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH,"models", f"model_{suffix}.h5")
        print(f"- model path: {model_path}")
        model.save(model_path)
        print(f'>Saved: {model_path}')

    print("\n✅ data saved locally")

    return None


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
