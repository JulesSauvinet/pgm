# Data describes patient medical record data for Pima Indians
# and whether they had an onset of diabetes within five years.
# It is a binary classification problem (onset of diabetes as 1 or not as 0)

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
print('Chargement des donnees')
dataframe = pd.read_csv("zoo.csv", header=0)
dataset = dataframe.values
X = dataset[:,1:17].astype(float)
Y = dataset[:,17]
print('...done\n')

# encode class values as integers
print('Encodage des classes')

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
Y = np_utils.to_categorical(Y)
print('...done\n')

# create model
model = Sequential()
model.add(Dense(16, input_dim=16, init='normal', activation='relu'))
model.add(Dense(8, init='normal', activation='sigmoid'))

# Compile model
# adam = the efficient gradient descent algorithm
# accuracy = we will collect and report the classification accuracy as the metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10,verbose=2)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.10, random_state = 42)

# evaluate the model
scores = model.evaluate(X_Test, Y_Test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

