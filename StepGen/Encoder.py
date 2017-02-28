import random
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, merge,Bidirectional
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers.noise import GaussianNoise

from MIPS import MIPS
from ARM import ARM


def test():
    while (True):
        try:
            inst = raw_input("Enter instruction: ")
            if inst == "QUIT":
                break
            elif inst == "SAVE":
                name=raw_input("Name: ")
                model.save(name+".h5")
                break
            inputs = np.array([encode_inst(inst)])
            inputs = sequence.pad_sequences(inputs, maxlen=20)
            predict = model.predict([inputs])
            for prediction in predict:
                print np.argmax(prediction),prediction
        except(Exception) as e:
            print e

train=raw_input("Train?")


num_regs = 15
symbols = list(map(chr, range(97, 123))) + map(str, range(0, 10)) + [",", " ", "#", "[", "]" ,"$",")","("] + ["*"]
print symbols
random.seed(12)
letters = list(map(chr,range(65,91)))
letters.remove('R')

def encode_char(c):
    '''if c == "R":
        v = random.choice(letters)
        v1 = random.random()
        if (v1 > 0.3):
            c = v'''
    l = [0] * len(symbols)
    l[symbols.index(c)] = 1
    return l

#This method has adds no noise to the input data
'''
def encode_char(c):
    l = [0] * len(symbols)
    l[symbols.index(c)] = 1
    return l
'''


def encode_inst(inst):
    encoded_inst = []
    num = False
    for c in inst:
        encoded_char = encode_char(c)
        encoded_inst.append(encoded_char)
        if c == "#":
            break
    return encoded_inst



op = "CMP"
ops = ["add","sub","b","mul","mov","cmp","str","ldr"]
conds = ["", "ne", "eq", "gt", "lt"]
arm = ARM()
mips = MIPS()
before = arm.total + mips.total

print "Samples generated"
random.shuffle(before)
samples = [x[0].lower() for x in before]
outputs = [x[1] for x in before]
print len(samples), len(outputs)
samples = samples
outputs = outputs


def split(l, ratio):
    train_data = l[:int(ratio * len(l))]
    test_Data = l[int(ratio * len(l)):]
    return train_data, test_Data


inputs = []
inst_outputs = []
cond_outputs = []
r0_outputs = []
r1_outputs = []
r2_outputs = []
const_outputs = []
branch_outputs = []
bracket_outputs = []
for sample in samples:
    inputs.append(encode_inst(sample))
for out in outputs:
    inst_outputs.append(out[0])
    r0_outputs.append(out[1])
    r1_outputs.append(out[2])
    r2_outputs.append(out[3])
    cond_outputs.append(out[4])
    bracket_outputs.append(out[5])
print len(inputs)
print len(inst_outputs)

ratio = 0.7
inputs, e_inputs = split(inputs, ratio)
inst_outputs, e_inst = split(inst_outputs, ratio)
r0_outputs, e_r0 = split(r0_outputs, ratio)
r1_outputs, e_r1 = split(r1_outputs, ratio)
r2_outputs, e_r2 = split(r2_outputs, ratio)
cond_outputs, e_outputs3 = split(cond_outputs, ratio)
bracket_outputs, e_bracket = split(bracket_outputs, ratio)

inputs = np.array(inputs)
inputs = sequence.pad_sequences(inputs, maxlen=20)
inst_outputs = np.array(inst_outputs)
r0_outputs = np.array(r0_outputs)
r1_outputs = np.array(r1_outputs)
r2_outputs = np.array(r2_outputs)
cond_outputs = np.array(cond_outputs)
bracket_outputs = np.array(bracket_outputs)

e_inputs = np.array(e_inputs)
e_inputs = sequence.pad_sequences(e_inputs, maxlen=20)
e_inst = np.array(e_inst)
e_r0 = np.array(e_r0)
e_r1 = np.array(e_r1)
e_r2 = np.array(e_r2)
e_conds = np.array(e_outputs3)
e_bracket = np.array(e_bracket)

if train=="Y":
    inputl = Input(shape=(20, len(symbols)))
    lstm1 = LSTM(512, activation='relu', name='lstm1', return_sequences=True)(inputl)
    lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False)(lstm1)
    lstm = GaussianNoise(0.1)(lstm)
    instr = Dense(len(ops), activation='softmax', name='instr')(lstm)
    cond = Dense(len(conds), activation='softmax', name='cond')(lstm)
    r0 = Dense(num_regs, activation="softmax", name='r0')(lstm)
    r1 = Dense(num_regs, activation="softmax", name='r1')(lstm)
    r2 = Dense(num_regs, activation="softmax", name='r2')(lstm)
    bracket = Dense(2, activation='softmax', name='bracket')(lstm)
    import winsound

    Freq = 2500  # Set Frequency To 2500 Hertz
    Dur = 1000
    model = Model(input=[inputl], output=[instr, cond, r0, r1, r2, bracket])
    model.compile(optimizer='rmsprop',
                  loss=['categorical_crossentropy'] * 6,
                  metrics=['accuracy'])
    print model.summary()


    while (True):
        num_epoch = int(raw_input("How many epochs? "))
        model.fit([inputs],
                  [inst_outputs, cond_outputs, r0_outputs, r1_outputs, r2_outputs, bracket_outputs],
                  batch_size=128,
                  nb_epoch=num_epoch)
        winsound.Beep(Freq, Dur)
        inp=raw_input("Test?")
        if inp=="Y":
            print model.evaluate([e_inputs], [e_inst, e_conds, e_r0, e_r1, e_r2, e_bracket])
        test()
        inp = raw_input("More epoch? Y?")
        if (inp != "Y"):
            break
else:
    model=load_model("Encoder_Mips_ARM_3.h5")
    test()
