from sklearn.model_selection import train_test_split
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
dataframe = pd.read_csv("voice.csv", header=0)
dataset = dataframe.values
X = dataset[:,0:20].astype(float)
Y2 = dataset[:,20]
print('...done\n')

print('Encodage des classes')
encoder = LabelEncoder()
Y = Y2
encoder.fit(Y)
Y = encoder.transform(Y)

StringM = 'male'
StringF = 'female'
IntM = 0
IntF = 0
cmpM = 0
cmpF = 0
for i in range(Y.shape[0]) :
    encodage = Y[i]
    reel = Y2[i]
    if (reel == 'male') :
        IntM = encodage
        cmpM += 1
    else :
        IntF = encodage
        cmpF += 1
    if ((cmpM != 0) & (cmpF != 0)) :
        break

print('Encodage du genre \'male\' par :',IntM)
print('Encodage du genre \'female\' par :',IntF)
print('...done\n')

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.10, random_state = 42)

# create model
model = Sequential()
model.add(Dense(20, input_dim=20, init='uniform', activation='relu'))
model.add(Dense(20, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
# binary_crossentropy = logarithmic loss
# adam = the efficient gradient descent algorithm
# accuracy = we will collect and report the classification accuracy as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_Train, Y_Train, nb_epoch=150, batch_size=10,verbose=2)

# evaluate the model
scores = model.evaluate(X_Test, Y_Test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X_Test)

# round predictions
rounded = [float(numpy.round(x)) for x in predictions]

Y_size = Y_Test.shape[0]
VP = 0
VN = 0
FP = 0
FN = 0
for i in range(Y_size) :
    reel = Y_Test[i,]
    predict = rounded[i]
    if (reel == predict) :
        if (reel == 0) :
            VN += 1
        else :
            VP += 1
    else :
        if (reel == 0) :
            FP += 1
        else :
            FN += 1
        
print ("Matrice de confusion")
print (" ____________________________________","\n"  \
         " P\R      Female    Male     ","\n"  \
         " ----------------------------------","\n"  \
         " Female ", " "*2, VP, " "*6, FP, " "*5, "","\n"  \
         " -----------------------------------","\n"  \
         " Male  ", " "*4, FN, " "*6, VN, " "*5, "","\n"  \
         " ___________________________________","\n")

correct = VP + VN
accuracy = (correct/Y_size)*100
print("%.2f%%" % (accuracy))
