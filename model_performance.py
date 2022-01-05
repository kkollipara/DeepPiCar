import os
import pickle

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

import constant
from train_model import image_data_generator
from train_model import load_data


def summarize_prediction(Y_true, Y_pred):
    mse = mean_squared_error(Y_true, Y_pred)
    r_squared = r2_score(Y_true, Y_pred)
    print(f'mse       = {mse:.2}')
    print(f'r_squared = {r_squared:.2%}')
    print()


def predict_and_summarize(X, Y, model_path):
    model = load_model(f'{model_path}')
    Y_pred = model.predict(X)
    summarize_prediction(Y, Y_pred)
    return Y_pred


def main():
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, constant.MODEL_FOLDER)
    model_path = os.path.join(model_dir, constant.MODEL_FINAL_NAME)
    history_path = os.path.join(model_dir, 'history.pickle')
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    print(history)
    plt.plot(history['loss'],color='blue')
    plt.plot(history['val_loss'],color='red')
    plt.legend(["training loss", "validation loss"])
    plt.show()

    n_tests = 500
    data_path = os.path.join(os.getcwd(), constant.DATA_FOLDER)
    image_paths, steering_angles = load_data(data_path)
    X_test, y_test = next(image_data_generator(image_paths, steering_angles, 100, False))

    y_pred = predict_and_summarize(X_test, y_test, model_path )

    n_tests_show = 3
    fig, axes = plt.subplots(n_tests_show, 1, figsize=(10, 4 * n_tests_show))
    for i in range(n_tests_show):
        axes[i].imshow(X_test[i])
        axes[i].set_title(f"actual angle={y_test[i]}, predicted angle={int(y_pred[i])}, diff = {int(y_pred[i]) - y_test[i]}")
    plt.show()

if __name__ == "__main__":
    main()