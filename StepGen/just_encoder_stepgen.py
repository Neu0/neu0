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



small_ops = ["READ", "WRITE", "ALU", "NO", "SET", "INCIN", "SETIN"]
weights = ["i", "c", "r0", "r1", "r2", ""]




model = load_model("Both_Model_str.h5")

num_regs = 15
open = real_open
while (True):
    instructions = [raw_input("Enter instruction: ")]
    encod = [encode_inst(i) for i in instructions]

    encodings = np.array(encod)
    encodings = sequence.pad_sequences(encodings, maxlen=20)

    model_predict = model.predict(encodings)
    for prediction in model_predict[:-2]:
        print np.argmax(prediction), prediction
    small_ops1 = model_predict[6][0]
    w = model_predict[7][0]
    small = ""
    print "-" * 15
    for op, w in zip(small_ops1, w):
        ind = np.argmax(op)
        ind2 = np.argmax(w)
        out = str(small_ops[ind]) + " " + str(weights[ind2])
        print out
