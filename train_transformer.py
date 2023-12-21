import os
import numpy
import mne
from tqdm import tqdm
import numpy as np
import bandpass_filter
import time_series_transformer as tst
import matplotlib.pyplot as plt
import transform_dataset as td
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


name_of_exp = "900s_901e_500samp_norm"

X, Y = td.transform(name_of_exp)

input_shape = (X.shape[1],X.shape[2])


def kfold(X, Y, k, seed, epochs, batch_size=None):

    # Split the data into k folds.
    np.random.seed(seed)

    # Shuffle the indices of the data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    X_folds = np.array_split(X_shuffled, k)
    Y_folds = np.array_split(Y_shuffled, k)

    # Create a dictionary to store the results.

    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "std_accuracy": [],
        "std_precision": [],
        "std_recall": [],
        "std_f1": [],
    }

    # Iterate over the k folds.

    for i in range(k):
        # Get the training and validation data for this fold.

        X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
        Y_train = np.concatenate([Y_folds[j] for j in range(k) if j != i])
        X_val = X_folds[i]
        Y_val = Y_folds[i]

        # Train the model on the training data.

        # Build the Transformer model
        model = tst.build_model(
            input_shape,
            head_size=256,
            num_heads=6,
            ff_dim=6,
            num_transformer_blocks=6,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],  # Change metric to accuracy for binary classification
        )
        model.load_weights(f'trained_weights/model_{name_of_exp}.h5') 
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

        # Evaluate the model on the validation data.

        prob_predictions = model.predict(X_val)
        threshold = 0.5
        predictions = np.where(prob_predictions > threshold, 1, 0)
        conf_matrix = confusion_matrix(Y_val, predictions)


        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)

        # Calculate precision
        precision = precision_score(Y_val, predictions)

        # Calculate recall
        recall = recall_score(Y_val, predictions)

        # Calculate F1-score
        f1 = f1_score(Y_val, predictions)

        print(f"{i+1} fold Results: {accuracy=}, {precision=}, {recall=}, {f1=}")
        # Append the values to your 'results' dictionary or list
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)

    # Calculate the mean and standard deviation of the results.

    results["mean_accuracy"] = np.mean(results["accuracy"])
    results["std_accuracy"] = np.std(results["accuracy"])
    results["mean_precision"] = np.mean(results["precision"])
    results["std_precision"] = np.std(results["precision"])
    results["mean_recall"] = np.mean(results["recall"])
    results["std_recall"] = np.std(results["recall"])
    results["mean_f1"] = np.mean(results["f1"])
    results["std_f1"] = np.std(results["f1"])

    return results

results = kfold(X, Y, k=5, seed = 42, epochs = 6, batch_size=1)

print(results)