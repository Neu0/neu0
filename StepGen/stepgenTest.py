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


ops= ["add", "sub", "b", "mul", "mov", "cmp", "str", "ldr"]
conds = ["", "ne", "eq", "gt", "lt"]

inputl = Input(shape=(20, len(symbols)))
lstm1 = LSTM(512, activation='relu', name='lstm1', return_sequences=True, weights=model.layers[1].get_weights())(inputl)
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False, weights=model.layers[2].get_weights())(lstm1)
new_model = Model(input=[inputl], output=[lstm])
new_model.compile("rmsprop", "mse")


small_ops = ["READ", "WRITE", "ALU", "NO", "SET","INCIN","SETIN"]
weights = ["i", "c", "r0", "r1", "r2", ""]

model = load_model("stepGenB_str.h5")
#model = load_model("stepGenB_str.h5")
num_regs = 15
lstm = Input(shape=(256,), name="encoding_inp")
instr = Dense(len(ops), activation='softmax', name='instr', weights=encoder_model.layers[4].get_weights())(lstm)
cond = Dense(len(conds), activation='softmax', name='cond', weights=encoder_model.layers[5].get_weights())(lstm)
r0 = Dense(num_regs, activation="softmax", name='r0', weights=encoder_model.layers[6].get_weights())(lstm)
r1 = Dense(num_regs, activation="softmax", name='r1', weights=encoder_model.layers[7].get_weights())(lstm)
r2 = Dense(num_regs, activation="softmax", name='r2', weights=encoder_model.layers[8].get_weights())(lstm)
bracket = Dense(2, activation='softmax', name='bracket', weights=encoder_model.layers[9].get_weights())(lstm)

exec_model = Model(input=[lstm], output=[instr, cond, r0, r1, r2, bracket])
exec_model.compile(optimizer='rmsprop',
                   loss=['categorical_crossentropy'] * 6,
                   metrics=['accuracy'])


open = real_open

instructions = ["STR R0,[R1]"]

instructions = [x.strip() for x in instructions]
instructions2 = [x.strip() for x in instructions]

encod = [encode_inst(i) for i in instructions]
inputs = np.array(encod)
inputs = sequence.pad_sequences(inputs, maxlen=20)
predict = new_model.predict([inputs])
encodings = np.array([predict])
f = open("bubble.neu","r")
#print encodings
slides=[]

while(True):
    #inst=raw_input("Enter Instruction")

    instructions = f.readlines()

    instructions = [x.strip() for x in instructions]
    instructions2 = [x.strip() for x in instructions]

    encod = [encode_inst(i) for i in instructions]
    inputs = np.array(encod)
    inputs = sequence.pad_sequences(inputs, maxlen=20)
    predict = new_model.predict([inputs])
    encodings = np.array([predict])
    smallMachine  = Small(5,15,exec_model,encodings[0])
    smallMachine.cache.disable()
    i=0
    result=[]
    res = []
    for z in range(10, 15):
        res.append(smallMachine.cache.memory.read(z))
        # print smallMachine.cache.memory.read(z)
    temp = copy.deepcopy(smallMachine.registers.get_memory())
    result.append([res, temp])
    count = 0
    while(round(smallMachine.ir)!=len(instructions)):
        slide = []
        temp = copy.deepcopy(smallMachine.registers.get_memory())
        memory_old=""
        for z in range(10,15):
            mem=smallMachine.cache.memory.read(z)
            res.append(mem)
            memory_old+="<td>"+str(mem)+"</td>"
        register_old=""
        for reg in temp:
            register_old+="<td>"+str(reg)+"</td>"
        #print smallMachine.encoding
        print instructions2[int(round(smallMachine.ir))]
        slide.append(instructions2[int(round(smallMachine.ir))])
        print "***"
        encodings = np.array([[smallMachine.encoding]])
        encodings = sequence.pad_sequences(encodings, maxlen=5, padding='post')
        #print encodings.shape
        model_predict = model.predict(encodings)
        small_ops1 = model_predict[0][0]
        w = model_predict[1][0]
        small=""
        for op, w in zip(small_ops1, w):
            ind = np.argmax(op)
            ind2 = np.argmax(w)
            small+="<p>"+str(small_ops[ind])+" "+str(weights[ind2])+"</p>"
            #small.append([small_ops[ind],small_ops[ind]])
            #print(small_ops[ind], weights[ind2])
            smallMachine.functions[small_ops[ind]](ind2)
            #smallMachine.show()
        slide.append(small)
        smallMachine.set_encoding()
        #smallMachine.show()
        res=[]
        memory_new = ""
        for z in range(10,15):
            mem = smallMachine.cache.memory.read(z)
            res.append(mem)
            memory_new += "<td>" + str(mem) + "</td>"
            #print smallMachine.cache.memory.read(z)
        temp = copy.deepcopy(smallMachine.registers.get_memory())
        register_new= ""
        for reg in temp:
            register_new += "<td>" + str(reg) + "</td>"
        result.append([res,temp])
        slide.append(register_old)
        slide.append(register_new)

        slide.append(memory_old)
        slide.append(memory_new)
        slides.append(slide)
        print "************"

        i+=1
        print i

    break
result=np.array(result)
print slides
np.save("iclr_result.npy",result)
np.save("slide.npy",result)
