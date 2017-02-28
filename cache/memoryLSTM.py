import random
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, merge,Bidirectional, Reshape
from keras.preprocessing import sequence
from keras.models import load_model

memorySize = 500
inputl = Input(shape=(memorySize,1))
lstm = LSTM(32,activation='tanh',return_sequences=True)(inputl)
op = Dense(1,activation='tanh')(lstm)
res = Reshape((memorySize,))(op)
final_op = Dense(memorySize,activation='softmax')(res)

model = Model(input=[inputl],output=[final_op])
model.compile("rmsprop","categorical_crossentropy",metrics=["accuracy"])
print model.summary()

l = [i for i in range(memorySize)]
before = []
for i in range(memorySize):
    target = [0]*memorySize
    target[i] = 1

    inp = [[(x-i)**2] for x in l]
    before.append([inp,target])
import random
random.shuffle(before)
samples = [x[0] for x in before]
targets = [x[1] for x in before]
samples = np.array(samples)
targets = np.array(targets)

model.fit(samples,targets,nb_epoch=50)

