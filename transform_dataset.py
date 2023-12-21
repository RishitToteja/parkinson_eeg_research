import numpy as np
import bandpass_filter


def transform(name_of_exp):

    path = "/content/drive/MyDrive/ML Research Project-EEG/Datasets/Numpy-Files/"

    train_control = np.load(path+f"train_control_{name_of_exp}.npy")
    train_pd = np.load(path+f"train_pd_{name_of_exp}.npy")

    X = np.concatenate([train_control, train_pd], axis=0)
    Y = np.array([0 for i in range(len(train_control))]+[1 for i in range(len(train_pd))])

    fs = 500
    lowcut = 1
    highcut = 49

    train_control_filtered = []
    train_pd_filtered = []

    for i in range(len(train_control)):

        signals = []
        for j in range(63):

            eeg_data = train_control[i][j]
            eeg_data_control_filtered = bandpass_filter.bbf(eeg_data, lowcut, highcut, fs)

            signals.append(eeg_data_control_filtered)

        signals = np.stack(signals, axis=0)
        train_control_filtered.append(signals)

    for i in range(len(train_pd)):

        signals = []
        for j in range(63):

            eeg_data = train_pd[i][j]
            eeg_data_pd_filtered = bandpass_filter.bbf(eeg_data, lowcut, highcut, fs)

        signals.append(eeg_data_pd_filtered)
        signals = np.stack(signals, axis=0)

    train_pd_filtered.append(signals)

    train_control_filtered = np.stack(train_control_filtered, axis=0)
    train_pd_filtered = np.stack(train_pd_filtered, axis=0)

    X = np.concatenate([train_control_filtered, train_pd_filtered], axis=0)
    Y = np.array([0 for i in range(len(train_control))]+[1 for i in range(len(train_pd))])

    return (X, Y)