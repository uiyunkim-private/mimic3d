import os
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
# from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, \
    LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, \
    StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, \
    StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from mlxtend.plotting import plot_learning_curves
from mlxtend.preprocessing import shuffle_arrays_unison
import keras
import keras.utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers,optimizers

print(os.getcwd())
print("Modules imported \n")
import os


def data_loader():



    # Load MIMIC2 data

    data = pd.read_csv('mimic3d.csv')
    print("With id", data.shape)
    data_full = data.drop('hadm_id', 1)
    print("No id", data_full.shape)

    print(data_full.shape)
    data_full.head(10)

    # Label = LOS

    y = data_full['LOSgroupNum']
    X = data_full.drop('LOSgroupNum', 1)
    X = X.drop('LOSdays', 1)
    X = X.drop('ExpiredHospital', 1)
    X = X.drop('AdmitDiagnosis', 1)
    X = X.drop('AdmitProcedure', 1)
    X = X.drop('marital_status', 1)
    X = X.drop('ethnicity', 1)
    X = X.drop('religion', 1)
    X = X.drop('insurance', 1)

    print("y - Labels", y.shape)
    print("X - No Label No id ", X.shape)
    print(X.columns)

    data_full.groupby('LOSgroupNum').size().plot.bar()
    plt.show()
    data_full.groupby('admit_type').size().plot.bar()
    plt.show()
    data_full.groupby('admit_location').size().plot.bar()
    plt.show()

    # MAP Text to Numerical Data
    # Use one-hot-encoding to convert categorical features to numerical

    print(X.shape)
    categorical_columns = [
        'gender',
        'admit_type',
        'admit_location'
    ]

    for col in categorical_columns:
        # if the original column is present replace it with a one-hot
        if col in X.columns:
            one_hot_encoded = pd.get_dummies(X[col])
            X = X.drop(col, axis=1)
            X = X.join(one_hot_encoded, lsuffix='_left', rsuffix='_right')

    print(X.shape)

    print(data_full.shape)
    print(X.shape)
    # XnotNorm = np.array(X.copy())
    XnotNorm = X.copy()
    print('XnotNorm ', XnotNorm.shape)

    ynotNorm = y.copy()
    print('ynotNorm ', ynotNorm.shape)

    # Normalize X

    x = XnotNorm.values  # returns a numpy array
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    XNorm = pd.DataFrame(x_scaled, columns=XnotNorm.columns)
    # print(XNorm)
    # print(y)
    print('X normalized')

    # SPLIT into Train & Test

    X_train, X_test, y_train, y_test = train_test_split(XNorm, y, test_size=0.2, random_state=7)
    print('X_train: ', X_train.shape)
    print('X_test: ', X_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape)
    # Transfer data to NN format

    x_val = X_test
    partial_x_train = X_train
    y_val = y_test
    partial_y_train = y_train

    print("partial_x_train ", partial_x_train.shape)
    print("partial_y_train ", partial_y_train.shape)

    print("x_val ", x_val.shape)
    print("y_val ", y_val.shape)

    yTrain = to_categorical(partial_y_train)
    yVal = to_categorical(y_val)
    print(yTrain.shape)
    print(yVal.shape)

    return partial_x_train, yTrain, x_val, yVal

def nn_model():
    model = Sequential()
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(30,)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    print(model.summary())

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def plot(history):
    # VALIDATION LOSS curves

    plt.clf()
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, (len(history_dict['loss']) + 1))
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # VALIDATION ACCURACY curves

    plt.clf()
    acc_values = history_dict['categorical_accuracy']
    val_acc_values = history_dict['val_categorical_accuracy']
    epochs = range(1, (len(history_dict['categorical_accuracy']) + 1))
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    partial_x_train, yTrain, x_val, yVal = data_loader()
    model = nn_model()
    NumEpochs = 10
    BatchSize = 64

    history = model.fit(partial_x_train, yTrain, epochs=NumEpochs, batch_size=BatchSize, validation_data=(x_val, yVal))

    results = model.evaluate(x_val, yVal)
    print("_" * 100)
    print("Test Loss and Accuracy")
    print("results ", results)
    history_dict = history.history
    history_dict.keys()
    plot(history)