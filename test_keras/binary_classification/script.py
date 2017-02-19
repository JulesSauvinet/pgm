import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# baseline model
def create_baseline():
    # create model
        # option 'normal' = weights init with a small Gaussian random number
        # option 'relu' = Rectifier activation function 
        # option 'sigmoid' = produce a probability output in the range of 0 to 1
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
        # option 'binary_crossentropy' = using the logarithmic loss function during training
        # option 'adam' = gradient descent
        # option 'accuracy' = collect the accurary when the model is training
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# smaller model
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
	model.add(Dense(30, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

    

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)

# load the dataset using pandas
print('Chargement des donnees')
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values

# and split the columns into 60 input variables (X) and 1 output variable (Y)
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
print('...done\n')

# encode class values as integers
print('Encodage des classes')
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print('...done\n')

# evaluate model with standardized dataset
print('Creation du model avec standardized dataset')
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print('...done\n')
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# evaluate baseline model with standardized dataset
print('Creation du baseline model avec standardized dataset')
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# evaluate model with smaller dataset
print('\nCreation du model avec smaller dataset')
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate model with larger dataset
print('\nCreation du model avec larger dataset')
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
