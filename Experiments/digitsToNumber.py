from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np
from keras.models import load_model, Model
import copy


length = 3
samples = [(map(lambda x: [int(x)], '0' * (length - len(str(i))) + str(i)))
           for i in range(0, 10 ** length)]
for ind, sample in enumerate(samples):
    op = [sample[0][0]]
    for i in sample[1:]:
        op.append(op[-1] * 10 + i[0])
    op = map(lambda x: [x], op)
    samples[ind] = (sample, op)

def to_one_hot(x):
    x = x[0]
    inp = [0]*10
    inp[x] = 1
    return inp
samples = [(map(lambda x: x,x[0]),x[1][-1]) for x in samples]
import random
random.shuffle(samples)
inp,target = zip(*samples)
inp = np.array(inp)
target = np.array(target)

input1 = Input(shape=(length,1))
lstm1 = Bidirectional(LSTM(512, activation='relu', name='lstm1', return_sequences=True))(input1)
lstm2 = Bidirectional(LSTM(256, activation='relu', name='lstm2', return_sequences=False))(lstm1)
output = Dense(1,activation='relu',name='output')(lstm2)
model = Model(input=[input1],output=[output])
model.compile(optimizer='rmsprop',
              loss=['mse'] * 1,
              metrics=['accuracy'])
model.fit([inp],[target],nb_epoch=25*2,batch_size=32)

while(True):
    num = raw_input("Enter the number")
    samples = [(map(lambda x: [int(x)], '0' * (length - len(str(i))) + str(i)))
               for i in [num]][0]
    inp = map(lambda x: x, samples)
    inp = np.array([inp])
    print model.predict(inp) #()

