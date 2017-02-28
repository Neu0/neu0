from Banks import *
from alu import ALU
import numpy as np


class Small:
    def __init__(self, out_size, reg_size, model, mem_size=1000):
        self.registers = RegisterBank("Registers", reg_size)
        self.registers.write(0, 10)
        self.registers.write(1, 20)
        self.registers.write(2, 30)
        self.registers.write(4, 40)
        self.memory = MemoryBank("main", mem_size)
        self.memory.write(40, 1000)
        self.memory.write(10, 10)
        self.memory.write(20, 3)
        # 0 : Inst,1:cond, 2:r0, 3:r1, 4:r2
        self.ops = [0 for i in range(out_size)]
        self.model = model
        self.ALU = ALU()
        self.functions = {}
        self.functions["READ"] = self.read
        self.functions["WRITE"] = self.write
        self.functions["ALU"] = self.alu
        self.functions["SET"] = self.sub_and_set
        self.functions["NO"] = self.noop
        self.flag = 0
        self.encoding = None

    def noop(self, extra=None):
        return None

    def set_encoding(self, encoding):
        self.encoding = encoding

    def sub_and_set(self, extra=None):
        sub = 1
        add = 0
        mul = 0
        result = self.ALU.compute([add, sub, mul], self.ops[3], self.ops[4])
        self.ops[2] = result
        self.flag = round(self.ops[2])

    def mem(self, weight_index=None):
        index = self.ops[3]
        value = self.memory.read_gauss(index)
        self.ops[3] = value

    def show(self):
        print self.flag
        print self.ops
        print self.registers.get_memory()

    def check(self, conds):
        max_ = np.argmax(conds)
        if (max_ == 0):
            return True
        if (max_ == 1):
            return self.flag != 0
        if (max_ == 2):
            return self.flag == 0
        if (max_ == 3):
            return self.flag > 0
        else:
            return self.flag < 0

    def read(self, weight_index):
        encoding = self.encoding
        values = self.model.predict(encoding)
        conds = values[1][0]
        if (not self.check(conds)):
            print "Not executing"
            return
        dist = values[weight_index][0]
        dist = self.sharpen(dist)
        value = self.registers.read_fuzzy(dist)
        self.ops[weight_index] = value
        if (weight_index == 3):
            bracket = values[-1][0]
            if (bracket[1] > bracket[0]):
                self.mem()

    def write(self, weight_index):
        encoding = self.encoding
        values = self.model.predict(encoding)
        conds = values[1][0]
        if (not self.check(conds)):
            print "Not executing"
            return

        dist = values[weight_index][0]
        dist = self.sharpen(dist)
        value = self.ops[weight_index]
        self.registers.write_fuzzy(dist, value)

    def alu(self, extra=None):
        encoding = self.encoding
        values = self.model.predict(encoding)
        conds = values[1][0]
        if (not self.check(conds)):
            print "Not executing"
            return

        dist = values[0][0]
        dist = self.sharpen(dist)
        sub = dist[0]
        add = dist[1]
        mul = dist[2]
        result = self.ALU.compute([add, sub, mul], self.ops[3], self.ops[4])
        self.ops[2] = result

    def sharpen(self, a, temperature=0.05):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return a
