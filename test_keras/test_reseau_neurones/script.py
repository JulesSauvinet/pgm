# Data describes patient medical record data for Pima Indians
# and whether they had an onset of diabetes within five years.
# It is a binary classification problem (onset of diabetes as 1 or not as 0)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.10, random_state = 42)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
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
print (" ________________________________________________", "\n"  \
         " P\R             Diabetique    Non-diabetique     ","\n"  \
         " ------------------------------------------------ ","\n"  \
         " Diabetique ", " "*7, VP, " "*12, FP, " "*10, "","\n"  \
         " ------------------------------------------------ ","\n"  \
         " Non-diabetique  ", " "*2, FN, " "*12, VN, " "*10, "","\n"  \
         " ________________________________________________ ","\n")

correct = VP + VN
accuracy = (correct/Y_size)*100
print("%.2f%%" % (accuracy))

