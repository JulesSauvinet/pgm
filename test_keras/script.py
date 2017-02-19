import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# ---- Define The Neural Network Model ----
def baseline_model():
    
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ---- Initialize Random Number Generator ----
seed = 18
numpy.random.seed(seed)

# ---- Load The Dataset ----
print('Chargement des donnees')
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
print('...done\n')


# ---- Encode The Output Variable ----

# encode class values as integers
print('Encodage des classes')
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print('...done\n')


# ---- Here, we pass the number of epochs as 200 and batch size as 5 to use when training the model ----
print('Creation du model')
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
print('...done\n')


# ---- Evaluate The Model with k-Fold Cross Validation ----
print('Evaluation')
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print('...done\n')
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


