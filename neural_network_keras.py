# This is a sample code of neural network with one hidden layer performed via keras,
# written from scratch by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is called Titanic.
# It contains several data of passengers such as sex, age, fare, and whether they survived or not.
# The dataset can be obtained from https://www.openml.org/search?type=data&sort=runs&id=40945

# The aim of this code is to estimate the parameters of logistic model
# and predict the survival outcome.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('titanic3.csv',index_col=False)

# Replacing text variables with binary values
# Replacing any unknown data with NaN
data.replace({'male': 1, 'female': 0}, inplace=True)
data.replace('?', np.nan, inplace= True)

# Dropping the rows that contain NaN and selecting the columns that will be used
data = data[['sex', 'pclass','age','fare','survived']].dropna()

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    (data.loc[:,data.columns != 'survived']),
     data.survived,
     test_size=0.2,
     random_state=0
     )

# Normalizing the data before processing
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
# Normalizing the test set after fitting with training set
# so that the model is not biased
x_test = sc.transform(x_test)

# Initializing the neural network model
model = Sequential()

# Adding layers to the blank neural network
# Size of first (input) layer matches the number of inputs.
model.add(Dense(4, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 4))
model.add(Dense(3, kernel_initializer = 'uniform', activation = 'tanh'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Preparing the neural network and training it
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50)

# Calculating the accuracy with test set
y_test_prediction = (model.predict(x_test) >= 0.5).astype(int)
print(metrics.accuracy_score(y_test, y_test_prediction))