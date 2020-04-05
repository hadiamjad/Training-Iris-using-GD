import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Excel file is already randomized.
# simply reading the excel file.
dfs = pd.read_excel("irisdataset.xlsx")
#print(dfs)

# splitting the dataset 50% for training anf 50% for testing
train, test = train_test_split(dfs, test_size=0.5)
#print(train)




