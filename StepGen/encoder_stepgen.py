from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np
from keras.regularizers import *
from keras.preprocessing import sequence
from keras.models import load_model, Model
import random
from new_small import Small
from keras.layers.noise import GaussianNoise
from MIPS import MIPS
from ARM import ARM

real_open = open





num_regs = 30
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


before = []
ops = ["add","sub","b","mul","mov","cmp","str","ldr"]
conds = ["", "ne", "eq", "gt", "lt"]
arm = ARM(num_regs)
mips = MIPS(num_regs)


def prepare_inputs_for_encoder(samples):
    inputs = []
    for sample,target in samples:
        inputs.append(encode_inst(sample))
    inputs = np.array(inputs)
    inputs = sequence.pad_sequences(inputs, maxlen=20)
    return inputs

#0 => Arithmetic, 1 => mov , 2 => str, 3 => branch, 4 => branchmips, 5 => cmp
inputs = [0]*6
inputs[0] = arm.add + mips.add
inputs[1] = arm.move + mips.load + arm.load
inputs[2] = arm.store + mips.store
inputs[3] = arm.branch + mips.branch
inputs[4] = mips.branch_condition
inputs[5] = arm.compare
encodings = [prepare_inputs_for_encoder(x) for x in inputs]
encodings  = [item for sublist in encodings  for item in sublist]

print "Inputs prepared"
#encodings = [new_model.predict(x) for x in inputs]

print "Encodings obtained"
small_ops = ["READ", "WRITE", "ALU", "NO", "SET","INCIN","SETIN"]
weights = ["i", "c", "r0", "r1", "r2", ""]

sequences = [
[["READ", "r2"], ["READ", "r1"], ["ALU", ""], ["WRITE", "r0"],["INCIN",""]],
[["READ", "r1"], ["WRITE", "r0"], ["NO", ""], ["NO", ""],["INCIN",""]],
[["READ", "r0"], ["READ", "r2"], ["WRITE", "r2"], ["NO", ""],["INCIN",""]],
[["READ", "r2"], ["SETIN",""], ["NO", ""], ["NO", ""], ["NO", ""]],
[["READ", "r1"], ["READ", "r0"],["SET", ""],["READ", "r2"], ["SETIN",""]],
[["READ", "r2"], ["READ", "r0"], ["NO", ""], ["SET", ""],["INCIN",""]]

]


def encode_op_wt(sequence):
    labels = []
    for step in sequence:
        inst = [0] * len(small_ops)
        inst[small_ops.index(step[0])] = 1
        w = [0] * len(weights)
        w[weights.index(step[1])] = 1
        labels.append(inst)
    return labels

def encode_w_wt(sequence):
    labels = []
    for step in sequence:
        inst = [0] * len(small_ops)
        inst[small_ops.index(step[0])] = 1
        w = [0] * len(weights)
        w[weights.index(step[1])] = 1
        labels.append(w)
    return labels

target_op = []
target_w = []
for index,encoding in enumerate(inputs):
    target_op = target_op + [[x for x in encode_op_wt(sequences[index])] for i in encoding]
    target_w = target_w + [[x for x in encode_w_wt(sequences[index])] for i in encoding]
print "Target op and w prepared"


outputs = []
for inp_type in inputs:
    for i in inp_type:
        outputs.append(i[1])


inst_outputs = []
r0_outputs = []
r1_outputs = []
r2_outputs = []
cond_outputs = []
bracket_outputs = []

for out in outputs:
    inst_outputs.append(out[0])
    r0_outputs.append(out[1])
    r1_outputs.append(out[2])
    r2_outputs.append(out[3])
    cond_outputs.append(out[4])
    bracket_outputs.append(out[5])


def split(l, ratio):
    train_data = l[:int(ratio * len(l))]
    test_Data = l[int(ratio * len(l)):]
    return train_data, test_Data


import random
big_data = zip(encodings,inst_outputs,r0_outputs,r1_outputs,r2_outputs,cond_outputs,bracket_outputs,target_op,target_w)
random.shuffle(big_data)
big_data = split(big_data,0.1)[0]
encodings,inst_outputs,r0_outputs,r1_outputs,r2_outputs,cond_outputs,bracket_outputs,target_op,target_w = zip(*big_data)
inst_outputs = np.array(inst_outputs)
r0_outputs = np.array(r0_outputs)
r1_outputs = np.array(r1_outputs)
r2_outputs = np.array(r2_outputs)
cond_outputs = np.array(cond_outputs)
bracket_outputs = np.array(bracket_outputs)
target_op = np.array(target_op)
target_w = np.array(target_w)
encodings = np.array(encodings)





print "New encodings prepared"

print "Building model and ready to train"

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
input_small = RepeatVector(5)(lstm)
lstm = LSTM(256, activation='relu', name='lstm2', return_sequences=True)(input_small)
lstm = GaussianNoise(0.1)(lstm)
instr_small = Dense(len(small_ops), activation='softmax', name='instr_small')(lstm)
W_small = Dense(len(weights), activation='softmax', name='W_small')(lstm)
model = Model(input=[inputl], output=[instr,cond,r0,r1,r2,bracket,instr_small,W_small])
model.compile(optimizer='rmsprop',
              loss=['categorical_crossentropy'] * 8,
              metrics=['accuracy'])
print model.summary()


def test_model():
    instructions = [raw_input("Enter instruction: ")]
    encod = [encode_inst(i) for i in instructions]
    encodings = np.array(encod)
    encodings = sequence.pad_sequences(encodings, maxlen=20, padding='post')
    predict = model.predict(encodings)
    print predict
    print "-"*30
    for prediction in predict[:-2]:
        print np.argmax(prediction), prediction
    small_ops1 = predict[6][0]
    w = predict[7][0]
    small = ""
    print "-" * 15
    for op, w in zip(small_ops1, w):
        ind = np.argmax(op)
        ind2 = np.argmax(w)
        out = str(small_ops[ind]) + " " + str(weights[ind2])
        print out

while(True):
    num_epoch = int(raw_input("Enter number of epochs to run: "))
    model.fit([encodings],
          [inst_outputs,cond_outputs,r0_outputs,r1_outputs,r2_outputs,bracket_outputs,target_op, target_w],
          batch_size=128,
          nb_epoch=num_epoch)
    cmd = raw_input("Enter Q to quit or T to test or S to save")
    if(cmd == "Q"):
        break
    if(cmd == "T"):
        test_model()
    if(cmd == "S"):
        name = raw_input("Enter name of the model")
        if(len(name) == 0):
            name = "Both_Model_30_str.h5"
        else:
            name = name + ".h5"
        model.save(name)
