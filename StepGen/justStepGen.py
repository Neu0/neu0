from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np
from keras.regularizers import *
from keras.preprocessing import sequence
from keras.models import load_model, Model
import random
from new_small import Small
import copy

real_open = open
model = load_model("Encoder_Mips_ARM_3.h5")
encoder_model = load_model("Encoder_Mips_ARM_3.h5")
inst = "ADD R0,R1,R2"
from keras import backend as K


def get_activations(model, X_batch):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    layer_output = get_3rd_layer_output([X_batch])
    return layer_output


num_regs = 15
symbols = list(map(chr, range(97, 123))) + map(str, range(0, 10)) + [",", " ", "#", "[", "]" ,"$",")","("] + ["*"]
print symbols


def encode_char(c):
    l = [0] * len(symbols)
    l[symbols.index(c)] = 1
    return l


def encode_inst(inst):
    inst = inst.lower()
    encoded_inst = []
    num = False
    for c in inst:
        encoded_char = encode_char(c)
        encoded_inst.append(encoded_char)
        if c == "#":
            break
    return encoded_inst



inputl = Input(shape=(20, len(symbols)))
lstm1 = LSTM(512, activation='relu', name='lstm1', return_sequences=True, weights=model.layers[1].get_weights())(inputl)
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False, weights=model.layers[2].get_weights())(lstm1)
new_model = Model(input=[inputl], output=[lstm])
new_model.compile("rmsprop", "mse")

small_ops = ["READ", "WRITE", "ALU", "NO", "SET", "INCIN", "SETIN"]
weights = ["i", "c", "r0", "r1", "r2", ""]




model = load_model("stepGenB_str.h5")

num_regs = 15
open = real_open
while (True):
    instructions = [raw_input("Enter instruction: ")]
    encod = [encode_inst(i) for i in instructions]
    inputs = np.array(encod) #(ALLExamples,number of timesteps,Dimension of timestep)
    inputs = sequence.pad_sequences(inputs, maxlen=20)
    predict = new_model.predict(inputs)

    encodings = np.array([predict])
    encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
    # print encodings.shape
    model_predict = model.predict(encodings)
    small_ops1 = model_predict[0][0]
    w = model_predict[1][0]
    small = ""
    print "-" * 15
    for op, w in zip(small_ops1, w):
        ind = np.argmax(op)
        ind2 = np.argmax(w)
        out = str(small_ops[ind]) + " " + str(weights[ind2])
        print out
