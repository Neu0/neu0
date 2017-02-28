
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np
from keras.regularizers import *
from keras.models import load_model

data=[]
m = 30
for i in range(m):
    target = [0]*m
    target[i] = 1
    noise = 0.01
    while(noise <=0.2):
        num = i-0.1 + noise
        data.append(([num],target))
        noise += 0.0005

import random
random.shuffle(data)
samples = [x[0] for x in data]
targets = [x[1] for x in data]

'''
model=Sequential()
model.add(Dense(32,input_dim=1,activation='relu'))
model.add(Reshape((32,1)))
model.add(Conv1D(15,3,activation='relu'))
model.add(MaxPooling1D(stride=3))

#model.add(Flatten())

#model.add(Reshape((3,1)))
model.add(Conv1D(15,2,activation='relu'))
model.add(MaxPooling1D(stride=3))

model.add(Conv1D(15,1,activation='relu'))
model.add(MaxPooling1D(stride=3))

model.add(Flatten())

model.add(Dense(16,activation='relu'))
model.add(Dense(m,activation='softmax'))
'''


model=Sequential()
model.add(Dense(32,input_dim=1,activation='relu',init='uniform'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(m,activation='softmax'))

model.compile("rmsprop","categorical_crossentropy",metrics=["accuracy"])
print model.summary()
model.fit(np.array(samples[:int(0.7*len(samples))]),np.array(targets[:int(0.7*len(samples))]),nb_epoch=60)
print model.evaluate(np.array(samples[int(0.7*len(samples)):]),np.array(targets[int(0.7*len(samples)):]))