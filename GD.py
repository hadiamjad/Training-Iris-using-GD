import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Reading file and seperating dependent and independent variables.
dataset = pd.read_excel("irisdataset.xlsx")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# data preprocessing: renaming the `Species` column
# encoding scheme: 0, 1, 2
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# splitting the dataset 50% for training anf 50% for testing
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# random weight vector
weight = np.random.rand(4, 1)






