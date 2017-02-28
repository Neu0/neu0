import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
class MemoryBank:
    def __init__(self,name,size):
        self.name=name
        self.size=size
        self.memory=[0]*size

    def read_fuzzy(self,weights):
        '''
        :param weights: numpy array with weights for each cell
        :return: value read fuzzily
        '''
        weights = weights.tolist()
        value = 0
        for weight,cell_value in zip(weights,self.memory):
            value += weight*cell_value
        return value


    def read(self,index):
        return self.memory[index]

    def write(self,index,value):
        self.memory[index] = value

    def get_memory(self):
        return self.memory

    def write_fuzzy(self,weights,value):
        """
        :param weights:  numpy array with weights for each cell
        :param value: value to write to each cell according to weights
        :return: None
        """
        weights = weights.tolist()
        #retain
        retain_weights = [1-x for x in weights]
        #We do not need erase vector
        for index in range(len(self.memory)):
            self.memory[index] = retain_weights[index] * self.memory[index]
        #Add
        for index in range(len(self.memory)):
            self.memory[index] += weights[index] * value

num_regs = 3
symbols = list(map(chr, range(65, 91))) + map(str, range(0, 10)) + [",", " ", "#", "[", "]"] + ["*"]
print symbols


def encode_char(c):
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

if __name__ == "__main__":
    registers = MemoryBank("Registers",3)
    weights = np.array([[1,0,0]])
    registers.write_fuzzy(weights[0],10)
    print registers.get_memory()
    model = load_model("encoder.h5")
    while(True):
        inst=raw_input("Inst: ")
        inputs = np.array([encode_inst(inst)])
        inputs = sequence.pad_sequences(inputs, maxlen=20)

        predict = model.predict([inputs])
        print predict
    weights = np.array([[0.999,5*10**-4,5*10**-4]])
    print registers.read_fuzzy(weights[0])

