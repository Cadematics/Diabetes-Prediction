from keras.models import Sequential
from keras.layers import Dense
import numpy as np
np.random.seed(7)

#Load data
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

#Define Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit Model
model.fit(X, Y, epochs=150, batch_size=10)

#Evaluate Model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
