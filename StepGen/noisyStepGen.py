from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np
from keras.regularizers import *
from keras.preprocessing import sequence
from keras.models import load_model, Model
import random
from new_small import Small

real_open = open
model = load_model("Encoder_noise_5.h5")
encoder_model = load_model("Encoder_noise_5.h5")
inst = "ADD R0,R1,R2"
from keras import backend as K


def get_activations(model, X_batch):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    layer_output = get_3rd_layer_output([X_batch])
    return layer_output


num_regs = 15
symbols = list(map(chr, range(65, 91))) + map(str, range(0, 10)) + [",", " ", "#", "[", "]"] + ["*"]
print symbols
letters = list(map(chr, range(65, 91)))
letters.remove('R')


def encode_char(c):
    if c == "R":
        v = random.choice(letters)
        v1 = random.random()
        if (v1 > 0.3):
            c = v
    l = [0] * len(symbols)
    l[symbols.index(c)] = 1
    return l


def encode_inst(inst):
    encoded_inst = []
    num = False
    for c in inst:
        encoded_char = encode_char(c)
        encoded_inst.append(encoded_char)
        if c == "#":
            break
    return encoded_inst


before = []
ops = ["SUB", "ADD", "MUL", "MOV", "CMP", "B", "STR"]
conds = ["", "NE", "EQ", "GT", "LT"]
sample = encode_inst(inst)
X_batch = np.array([sample])
X_batch = sequence.pad_sequences(X_batch, maxlen=20)
for op in ops[:3]:
    for cond in conds:  # 5
        for r0 in range(0, num_regs - 1):  # 14
            for bracket in [True, False]:  # 2
                open = close = ""
                i_bracket = [1, 0]
                if bracket:
                    open = "["
                    close = "]"
                    i_bracket = [0, 1]
                for r1 in range(0, num_regs - 1):  # 14
                    for r2 in range(0, num_regs - 1):  # 14 : 27,440
                        if (r0 == r1 or r0 == r2):
                            continue
                        inst = op + cond + " R" + str(r0) + "," + open + "R" + str(r1) + close + ",R" + str(r2)
                        i_r2 = [0] * num_regs
                        i_r2[r2] = 1
                        # samples.append(inst)
                        i = [0] * len(ops)
                        i[ops.index(op)] = 1
                        i_cond = [0] * len(conds)
                        i_cond[conds.index(cond)] = 1
                        i_r0 = [0] * num_regs
                        i_r0[r0] = 1
                        i_r1 = [0] * num_regs
                        i_r1[r1] = 1
                        # outputs.append([i,i_r0,i_r1,i_r2])
                        branch_addr = 0
                        before.append((inst, [i, i_r0, i_r1, i_r2, i_cond, i_bracket]))
op = "MOV"
before2 = []
for r0 in range(num_regs - 1):  # 14
    for cond in conds:  # 5
        for j in range(15):  # 15
            for r2 in range(0, num_regs - 1):  # 14
                inst = op + cond + " R" + str(r0) + "," + "R" + str(r2)
                i_r2 = [0] * num_regs
                i_r2[r2] = 1
                i_r1 = [0] * num_regs
                i_r1[-1] = 1
                i_r0 = [0] * num_regs
                i_r0[r0] = 1
                i = [0] * len(ops)
                i[ops.index(op)] = 1
                i_cond = [0] * len(conds)
                i_cond[conds.index(cond)] = 1
                before2.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [1, 0]]))

# TODO- Create conditions for mov and adjust the dataset size
# For CMP - Dataset Creation
op = "CMP"
before3 = []
for j in range(80):
    for r0 in range(num_regs - 1):
        for r2 in range(0, num_regs - 1):
            inst = op + " R" + str(r0) + "," + "R" + str(r2)
            i_r2 = [0] * num_regs
            i_r2[r2] = 1
            i_r1 = [0] * num_regs
            i_r1[-1] = 1
            i_r0 = [0] * num_regs
            i_r0[r0] = 1
            i = [0] * len(ops)
            i[ops.index(op)] = 1
            i_cond = [0] * len(conds)
            i_cond[conds.index("")] = 1
            before3.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [1, 0]]))
before4 = []
# For branch -Dataset creation
for j in range(5):
    for cond in conds:  # 5
        for r in range(num_regs - 1):  # 14
            inst = "B" + cond + " R" + str(r)
            r_out = [0] * num_regs
            r_out[-1] = 1
            r2_out = [0] * num_regs
            r2_out[r] = 1
            i_inst = [0] * len(ops)
            i_inst[ops.index("B")] = 1
            i_cond = [0] * len(conds)
            i_cond[conds.index(cond)] = 1
            before4.append((inst, [i_inst, r_out, r_out, r2_out, i_cond, [1, 0]]))

# For STR -Dataset creation
before5 = []
op = "STR"
for r0 in range(num_regs - 1):  # 14
    for cond in conds:  # 5
        for j in range(15):  # 15
            for r2 in range(0, num_regs - 1):  # 14
                inst = op + cond + " R" + str(r0) + "," + "[R" + str(r2) + "]"
                i_r2 = [0] * num_regs
                i_r2[r2] = 1
                i_r1 = [0] * num_regs
                i_r1[-1] = 1
                i_r0 = [0] * num_regs
                i_r0[r0] = 1
                i = [0] * len(ops)
                i[ops.index(op)] = 1
                i_cond = [0] * len(conds)
                i_cond[conds.index(cond)] = 1
                before5.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [0, 1]]))

print "Here"
random.shuffle(before)
random.shuffle(before2)
random.shuffle(before3)
random.shuffle(before4)
random.shuffle(before5)
samples = [x[0] for x in before]
samples2 = [x[0] for x in before2]
samples3 = [x[0] for x in before3]
samples4 = [x[0] for x in before4]
samples5 = [x[0] for x in before5]


def split(l, ratio):
    train_data = l[:int(ratio * len(l))]
    test_Data = l[int(ratio * len(l)):]
    return train_data, test_Data


inputs = []
inputs2 = []
inputs3 = []
inputs4 = []
inputs5 = []
for sample in samples:
    inputs.append(encode_inst(sample))

for sample in samples2:
    inputs2.append(encode_inst(sample))

for sample in samples3:
    inputs3.append(encode_inst(sample))

for sample in samples4:
    inputs4.append(encode_inst(sample))

for sample in samples5:
    inputs5.append(encode_inst(sample))

inputs = np.array(inputs)
inputs = sequence.pad_sequences(inputs, maxlen=20)
inputs2 = np.array(inputs2)
inputs2 = sequence.pad_sequences(inputs2, maxlen=20)
inputs3 = np.array(inputs3)
inputs3 = sequence.pad_sequences(inputs3, maxlen=20)
inputs4 = np.array(inputs4)
inputs4 = sequence.pad_sequences(inputs4, maxlen=20)
inputs5 = np.array(inputs5)
inputs5 = sequence.pad_sequences(inputs5, maxlen=20)

inputl = Input(shape=(20, len(symbols)))
lstm1 = LSTM(512, activation='relu', name='lstm1', return_sequences=True, weights=model.layers[1].get_weights())(inputl)
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False, weights=model.layers[2].get_weights())(lstm1)
new_model = Model(input=[inputl], output=[lstm])
new_model.compile("rmsprop", "mse")
encodings = new_model.predict(inputs)
encodings2 = new_model.predict(inputs2)
encodings3 = new_model.predict(inputs3)
encodings4 = new_model.predict(inputs4)
encodings5 = new_model.predict(inputs5)

small_ops = ["READ", "WRITE", "ALU", "NO", "SET", "INCIN", "SETIN"]
weights = ["i", "c", "r0", "r1", "r2", ""]

"ADD"  "[read,r2],[read,r1],[alu,],[write,r0]"
sequence1 = [["READ", "r2"], ["READ", "r1"], ["ALU", ""], ["WRITE", "r0"], ["INCIN", ""]]
sequence2 = [["READ", "r2"], ["WRITE", "r0"], ["NO", ""], ["NO", ""], ["INCIN", ""]]
sequence3 = [["READ", "r2"], ["READ", "r0"], ["NO", ""], ["SET", ""], ["INCIN", ""]]
sequence4 = [["READ", "r2"], ["SETIN", ""], ["NO", ""], ["NO", ""], ["NO", ""]]
sequence5 = [["READ", "r0"], ["READ", "r2"], ["WRITE", "r2"], ["NO", ""], ["INCIN", ""]]


def encode(sequence):
    labels = []
    for step in sequence:
        inst = [0] * len(small_ops)
        inst[small_ops.index(step[0])] = 1
        w = [0] * len(weights)
        w[weights.index(step[1])] = 1
        labels.append([inst, w])
    return labels


target_op = [[x[0] for x in encode(sequence1)] for i in encodings]
target_op = target_op + [[x[0] for x in encode(sequence2)] for i in encodings2]
target_op = target_op + [[x[0] for x in encode(sequence3)] for i in encodings3]
target_op = target_op + [[x[0] for x in encode(sequence4)] for i in encodings4]
target_op = target_op + [[x[0] for x in encode(sequence5)] for i in encodings5]
target_w = [[x[1] for x in encode(sequence1)] for i in encodings]
target_w = target_w + [[x[1] for x in encode(sequence2)] for i in encodings2]
target_w = target_w + [[x[1] for x in encode(sequence3)] for i in encodings3]
target_w = target_w + [[x[1] for x in encode(sequence4)] for i in encodings4]
target_w = target_w + [[x[1] for x in encode(sequence5)] for i in encodings5]
target_op = np.array(target_op)
target_w = np.array(target_w)
new_encoding = []
for encoding in encodings:  # (60)
    new_encoding.append(np.array([encoding]))

for encoding in encodings2:  # (60)
    new_encoding.append(np.array([encoding]))

for encoding in encodings3:  # (60)
    new_encoding.append(np.array([encoding]))

for encoding in encodings4:  # (60)
    new_encoding.append(np.array([encoding]))

for encoding in encodings5:  # (60)
    new_encoding.append(np.array([encoding]))

encodings = np.array(new_encoding)
encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
print encodings.shape
input_small = Input(shape=(5, 256), name='input_small')
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=True)(input_small)
instr = Dense(len(small_ops), activation='softmax', name='instr')(lstm)
W = Dense(len(weights), activation='softmax', name='W')(lstm)
model = Model(input=[input_small], output=[instr, W])
model.compile(optimizer='rmsprop',
              loss=['categorical_crossentropy'] * 2,
              metrics=['accuracy'])
model.fit([encodings],
          [target_op, target_w],
          batch_size=128,
          nb_epoch=6)
model.save("stepGenNoise_1.h5")
# model = load_model("stepGenB_str.h5")
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
f = open("bubble.neu", "r")
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
# print encodings

smallMachine = Small(5, 15, exec_model, encodings[0])
i = 0
while (round(smallMachine.ir) != len(instructions)):
    if i == 50:
        break
    # print smallMachine.encoding
    print instructions2[int(round(smallMachine.ir))]
    print "***"
    encodings = np.array([[smallMachine.encoding]])
    encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
    # print encodings.shape
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
    for z in range(10, 15):
        print smallMachine.cache.memory.read(z)
    print "************"
    i += 1
    print i
