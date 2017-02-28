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


before = []
ops = ["SUB", "ADD", "MUL", "MOV", "CMP", "B","STR"]
conds = ["", "NE", "EQ", "GT", "LT"]

arm = ARM()
mips = MIPS()


def prepare_inputs_for_encoder(samples):
    inputs = []
    for sample,target in samples:
        inputs.append(encode_inst(sample))
    inputs = np.array(inputs)
    inputs = sequence.pad_sequences(inputs, maxlen=20)
    return inputs

inputl = Input(shape=(20, len(symbols)))
lstm1 = LSTM(512, activation='relu', name='lstm1', return_sequences=True, weights=model.layers[1].get_weights())(inputl)
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False, weights=model.layers[2].get_weights())(lstm1)
new_model = Model(input=[inputl], output=[lstm])
new_model.compile("rmsprop", "mse")
#0 => Arithmetic, 1 => mov , 2 => str, 3 => branch, 4 => branchmips, 5 => cmp
inputs = [0]*6
inputs[0] = arm.add + mips.add
inputs[1] = arm.move + mips.load + arm.load
inputs[2] = arm.store + mips.store
inputs[3] = arm.branch + mips.branch
inputs[4] = mips.branch_condition
inputs[5] = arm.compare
inputs = [prepare_inputs_for_encoder(x) for x in inputs]
print "Inputs prepared"
encodings = [new_model.predict(x) for x in inputs]
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
for index,encoding in enumerate(encodings):
    target_op = target_op + [[x for x in encode_op_wt(sequences[index])] for i in encoding]
    target_w = target_w + [[x for x in encode_w_wt(sequences[index])] for i in encoding]
print "Target op and w prepared"
target_op = np.array(target_op)
target_w = np.array(target_w)
new_encoding = []
for encoding_type in encodings:  # (60)
    for encoding in encoding_type:
        new_encoding.append(np.array([encoding]))



print "New encodings prepared"
encodings = np.array(new_encoding)
encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
print encodings.shape
print "Building model and ready to train"
input_small = Input(shape=(5, 256), name='input_small')
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=True)(input_small)
lstm = GaussianNoise(0.1)(lstm)
instr = Dense(len(small_ops), activation='softmax', name='instr')(lstm)
W = Dense(len(weights), activation='softmax', name='W')(lstm)
model = Model(input=[input_small], output=[instr, W])
model.compile(optimizer='rmsprop',
              loss=['categorical_crossentropy'] * 2,
              metrics=['accuracy'])
model.fit([encodings],
          [target_op, target_w],
          batch_size=128,
          nb_epoch=4)
model.save("stepGenB_str.h5")
#model = load_model("stepGenB_str.h5")
num_regs = 15
lstm = Input(shape=(256,), name="encoding_inp")
instr = Dense(len(ops), activation='softmax', name='instr', weights=encoder_model.layers[3].get_weights())(lstm)
cond = Dense(len(conds), activation='softmax', name='cond', weights=encoder_model.layers[4].get_weights())(lstm)
r0 = Dense(num_regs, activation="softmax", name='r0', weights=encoder_model.layers[5].get_weights())(lstm)
r1 = Dense(num_regs, activation="softmax", name='r1', weights=encoder_model.layers[6].get_weights())(lstm)
r2 = Dense(num_regs, activation="softmax", name='r2', weights=encoder_model.layers[7].get_weights())(lstm)
bracket = Dense(2, activation='softmax', name='bracket', weights=encoder_model.layers[8].get_weights())(lstm)

exec_model = Model(input=[lstm], output=[instr, cond, r0, r1, r2, bracket])
exec_model.compile(optimizer='rmsprop',
                   loss=['categorical_crossentropy'] * 6,
                   metrics=['accuracy'])

# smallMachine = Small(5, 15, exec_model)
# while (True):
#     inst = raw_input("Enter inst: ")
#     if inst == "SHOW":
#         smallMachine.show()
#         continue
#     inputs = np.array([encode_inst(inst)])
#     inputs = sequence.pad_sequences(inputs, maxlen=20)
#     predict = new_model.predict([inputs])
#     encodings = np.array([predict])
#     print exec_model.predict(encodings[0])
#     smallMachine.set_encoding(encodings[0])
#     encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
#     model_predict = model.predict(encodings)
#     small_ops1 = model_predict[0][0]
#     w = model_predict[1][0]
#     for op, w in zip(small_ops1, w):
#         ind = np.argmax(op)
#         ind2 = np.argmax(w)
#         print(small_ops[ind], weights[ind2])
#         #smallMachine.functions[small_ops[ind]](ind2)

open = real_open
f = open("bubble.neu","r")
instructions = f.readlines()
instructions = [x.strip() for x in instructions]
instructions2 = [x.strip() for x in instructions]
# instructions = [
#         #"ADD R6,R1,R2","SUB R7,R2,R3","MUL R8,[R1],R2"
# "ADD R5,R0,R1",
# "CMP R0,R1",
# "SUB R0,R0,R1",
# "BGT R2",
# "STR R5,[R1]"
# ]
encod = [encode_inst(i) for i in instructions]
inputs = np.array(encod)
inputs = sequence.pad_sequences(inputs, maxlen=20)
predict = new_model.predict([inputs])
encodings = np.array([predict])
#print encodings

smallMachine  = Small(5,15,exec_model,encodings[0])
i=0
while(round(smallMachine.ir)!=len(instructions)):
    if i==50:
        break
    #print smallMachine.encoding
    print instructions2[int(round(smallMachine.ir))]
    print "***"
    encodings = np.array([[smallMachine.encoding]])
    encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
    #print encodings.shape
    model_predict = model.predict(encodings)
    small_ops1 = model_predict[0][0]
    w = model_predict[1][0]
    for op, w in zip(small_ops1, w):
        ind = np.argmax(op)
        ind2 = np.argmax(w)
        print(small_ops[ind], weights[ind2])
        smallMachine.functions[small_ops[ind]](ind2)
        smallMachine.show()
    smallMachine.set_encoding()
    smallMachine.show()
    for z in range(10,15):
        print smallMachine.cache.memory.read(z)
    print "************"
    i+=1
    print i
