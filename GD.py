import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def dataPreprocessing():
    # Reading file and seperating dependent and independent variables.
    dataset = pd.read_excel("irisdataset.xlsx")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    # data preprocessing: renaming the `Species` column
    # encoding scheme: 0, 1, 2
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # adding the bias in the X dataset
    X = np.insert(X, 0, values=1, axis=1)

    # splitting the dataset 50% for training anf 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    y_train = np.reshape(y_train, (75, 1))
    y_test = np.reshape(y_test, (75, 1))

    # random weight vector
    weight = np.random.rand(5, 1)


    return X_train, X_test, y_train, y_test, weight

# cost function
def cost_func(weights, X, y):
    yhat = np.dot(X, weights)
    c = (1/2) * np.sum(np.square(y-yhat))
    return c

# gradient descent function
def gradient_desc(weights, X, y, training_rate = 0.0001, iterations = 1000):
    yhat = np.dot(X, weights)
    cost_document = []
    cost = -1
    i = 0
    while(cost == 0 or i < iterations):
        weights = np.subtract(weights, (-1) * training_rate * (np.dot(X.T, (y-yhat))))
        yhat = np.dot(X, weights)
        cost = cost_func(weights, X, y)
        if(i % 100 == 0):
            cost_document.append(cost)
        i = i + 1
    return weights, cost_document

# calculate mismatches
def calc_mismatches(weights, X, y):
    yhat = np.dot(X, weights)
    yhat = step_func(yhat)
    return np.count_nonzero(y-yhat)


# step function
def step_func(yhat):
    yhat[yhat <= 0.5] = 0
    yhat[yhat > 1.5] = 2
    yhat[(yhat > 0.5) & (yhat <= 1.5)] = 1
    return yhat


# main function
def __main__():
   X_train, X_test, y_train, y_test, weights = dataPreprocessing()
   cost_document = []

   weights, cost_document = gradient_desc(weights, X_train, y_train)
   print(calc_mismatches(weights, X_test, y_test))

   y = np.arange(start=1, stop=len(cost_document)+1, step=1)
   print(cost_document)
   plt.plot(y, cost_document, 'ro')
   plt.show()


# calling main function
__main__()
