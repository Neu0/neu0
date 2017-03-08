from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
import numpy as np
from keras.regularizers import *
from keras.preprocessing import sequence
from keras.models import load_model, Model
import random
from decimal import Decimal
from new_small import Small
import copy

def getMemoryContents(smallMachine):
    start = 10
    end = 30
    result= []
    if(smallMachine.cache.memory.read(15) < 0.5):
        for z in range(10,15):
            mem = smallMachine.cache.memory.read(z)
            result.append(mem)
        return reversed(result)
    else:
        append = False
        c = 5
        ind = -1
        for i in range(end,start-1,-1):
            if smallMachine.cache.memory.read(i)>0.5:
                ind = i
        for j in range(ind-4,ind+1):
            mem = smallMachine.cache.memory.read(j)
            result.append(mem)
        return  reversed(result)
real_open = open



num_regs = 15
symbols = list(map(chr, range(97, 123))) + map(str, range(0, 10)) + [",", " ", "#", "[", "]" ,"$",")","("] + ["*"]
print symbols
nums_strs = list(map(str,range(0,10)))


def encode_char(c):
    l = [0] * len(symbols)
    l[symbols.index(c)] = 1
    return l


def encode_inst(inst):
    inst = inst.lower()
    encoded_inst = []
    num = False
    const = 0
    for index,c in enumerate(inst):
        encoded_char = encode_char(c)
        encoded_inst.append(encoded_char)
        if index>0 and inst[index-1] == "," and c in nums_strs:
            const = int(inst[index:])
            break
        if c == "#":
            const = int(inst[index+1:])
            break
    return encoded_inst,const


small_ops = ["READ", "WRITE", "ALU", "NO", "SET", "INCIN", "SETIN"]
weights = ["i", "c", "r0", "r1", "r2", ""]





model = load_model("Full_Encoder.h5")
encoder_model = model
ops= ["add", "sub", "b", "mul", "mov", "cmp", "str", "ldr"]
conds = ["", "ne", "eq", "gt", "lt"]

inputl = Input(shape=(20, len(symbols)))
lstm1 = Bidirectional(LSTM(512, activation='relu', name='lstm1', return_sequences=True),weights=model.layers[1].get_weights())(inputl)
lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False, weights=model.layers[2].get_weights())(lstm1)
new_model = Model(input=[inputl], output=[lstm])
new_model.compile("rmsprop", "mse")

num_regs = 15
lstm = Input(shape=(256,), name="encoding_inp")
instr = Dense(len(ops), activation='softmax', name='instr', weights=encoder_model.layers[7].get_weights())(lstm)
cond = Dense(len(conds), activation='softmax', name='cond', weights=encoder_model.layers[8].get_weights())(lstm)
r0 = Dense(num_regs, activation="softmax", name='r0', weights=encoder_model.layers[9].get_weights())(lstm)
r1 = Dense(num_regs, activation="softmax", name='r1', weights=encoder_model.layers[10].get_weights())(lstm)
r2 = Dense(num_regs, activation="softmax", name='r2', weights=encoder_model.layers[11].get_weights())(lstm)
bracket = Dense(2, activation='softmax', name='bracket', weights=encoder_model.layers[12].get_weights())(lstm)

exec_model = Model(input=[lstm], output=[instr, cond, r0, r1, r2, bracket])
exec_model.compile(optimizer='rmsprop',
                   loss=['categorical_crossentropy'] * 6,
                   metrics=['accuracy'])

inp = Input(shape=(256,))
input_small = RepeatVector(5)(inp)
lstm = LSTM(256, activation='relu', name='lstm2', return_sequences=True,weights=encoder_model.layers[5].get_weights())(input_small)
instr_small = Dense(len(small_ops), activation='softmax', name='instr_small',weights=encoder_model.layers[13].get_weights())(lstm)
W_small = Dense(len(weights), activation='softmax', name='W_small',weights=encoder_model.layers[14].get_weights())(lstm)
step_model = Model(input=[inp], output=[instr_small,W_small])
step_model.compile(optimizer='adadelta',
              loss=['categorical_crossentropy'] * 2,
              metrics=['accuracy'])


num_regs = 15
open = real_open
f = open("fact.neu","r")
slides = []
while(True):
    #inst=raw_input("Enter Instruction")

    instructions = f.readlines()

    instructions = [x.strip() for x in instructions]
    instructions2 = [x.strip() for x in instructions]

    encod = [encode_inst(i) for i in instructions]
    encod, consts = zip(*encod)
    inputs = np.array(encod)
    inputs = sequence.pad_sequences(inputs, maxlen=20)
    predict = new_model.predict([inputs])
    encodings = np.array([predict])
    smallMachine  = Small(5,15,exec_model,encodings[0],consts)
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
    while(round(smallMachine.ir)!=len(instructions) and count<350):
        slide = []
        temp = copy.deepcopy(smallMachine.registers.get_memory())
        memory_old=""
        for mem in getMemoryContents(smallMachine):
            res.append(mem)
            memory_old+="<td>"+str(round(Decimal(str(mem)), 2))+"</td>"
        register_old=""
        for reg in temp:
            register_old+="<td>"+str(round(Decimal(str(reg)), 2))+"</td>"
        #print smallMachine.encoding
        print instructions2[int(round(smallMachine.ir))]
        slide.append(instructions2[int(round(smallMachine.ir))])
        print "***"
        encodings = np.array([smallMachine.encoding])
        #print encodings.shape
        model_predict = step_model.predict(encodings)
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

        for mem in getMemoryContents(smallMachine):
            res.append(mem)
            memory_new += "<td>" + str(round(Decimal(str(mem)), 2))+ "</td>"
            #print smallMachine.cache.memory.read(z)
        temp = copy.deepcopy(smallMachine.registers.get_memory())
        register_new= ""
        for reg in temp:
            register_new += "<td>" + str(round(Decimal(str(reg)), 2))+ "</td>"
        result.append([res,temp])
        slide.append(register_old)
        slide.append(register_new)

        slide.append(memory_old)
        slide.append(memory_new)
        slides.append(slide)
        print "************"
        count+=1
        i+=1
        print i

    break
result=np.array(result)
print slides
np.save("iclr_result.npy",result)
np.save("slide.npy",result)
