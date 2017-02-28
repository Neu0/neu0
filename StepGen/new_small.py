from Banks import *
from alu import ALU
import numpy as np


class Small:
    def __init__(self, out_size, reg_size, model, insts,mem_size=1000):
        self.registers = RegisterBank("Registers", reg_size)
        self.registers.write(0, 10)
        self.registers.write(1, 5)
        self.registers.write(2, 1)
        self.registers.write(3, 0)
        self.registers.write(10, 0)
        self.registers.write(13, 17)
        self.registers.write(6, 4)

        self.insts = MemoryBank("insts",len(insts))
        for index,encoding in enumerate(insts):
            self.insts.write(index,encoding)
        self.ir = 0
        self.cache =CacheBank()
        self.cache.memory.write(10, 2)
        self.cache.memory.write(11, 1)
        self.cache.memory.write(12, 9)
        self.cache.memory.write(13, 12)
        self.cache.memory.write(14, 4)
        self.operations = ["add","sub","b","mul","mov","cmp","str","ldr"]
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
        self.functions["INCIN"] = self.inc_in
        self.functions["SETIN"] = self.set_in
        self.flag = 0
        self.encoding = self.insts.read_gauss(self.ir)


    def noop(self, extra=None):
        return None

    def set_encoding(self):
        self.encoding = self.insts.read_gauss(self.ir)

    def sub_and_set(self, extra=None):
        sub = 1
        add = 0
        mul = 0
        result = self.ALU.compute([add, sub, mul], self.ops[2], self.ops[4])
        self.ops[2] = result
        self.flag = round(self.ops[2])

    def inc_in(self,extra =None):
        self.ir += 1

    def set_in(self,extra = None):
        encoding = self.encoding
        # print "SMALL",encoding.shape,encoding
        print "Inside setin"
        values = self.model.predict(np.array([encoding]))
        conds = values[1][0]
        print conds
        if (not self.check(conds)):
            print "Not executing"
            self.ir += 1
            return
        self.ir = self.ops[4]

    def mem(self, weight_index=None):
        index = self.ops[3]
        value = self.cache.read(index)
        self.ops[3] = value

    def printList(self,l, name,start=0):
        print name
        print " " + "_" * 25
        for i, v in enumerate(l):
            print "|" + " ".center(25, ' ') + "|"
            topr = str(i) + " : " + str(v)
            print "|" + topr.center(25, ' ') + "|"
            print "|_" + "_".center(23, '_') + "_|"
        print

    def show(self):
        self.printList([self.flag],"FLAGS")
        self.printList(self.ops,"OPS")
        self.printList(self.registers.get_memory(),"REGS")

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
        #print "SMALL",encoding.shape,encoding
        values = self.model.predict(np.array([encoding]))
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
        values = self.model.predict(np.array([encoding]))
        conds = values[1][0]
        if (not self.check(conds)):
            print "Not executing"
            return

        if (np.argmax(values[0][0]) == self.operations.index("mov")):
            self.ops[2] = self.ops[3]

        if (np.argmax(values[0][0]) == self.operations.index("str")):
                self.cache.write(self.ops[4], self.ops[2])
        else:
            dist = values[weight_index][0]
            dist = self.sharpen(dist)
            value = self.ops[weight_index]
            self.registers.write_fuzzy(dist, value)

    def alu(self, extra=None):
        encoding = self.encoding
        values = self.model.predict(np.array([encoding]))
        conds = values[1][0]
        if (not self.check(conds)):
            print "Not executing"
            return

        dist = values[0][0]
        dist = self.sharpen(dist)
        sub = dist[1]
        add = dist[0]
        mul = dist[2]
        result = self.ALU.compute([add, sub, mul], self.ops[3], self.ops[4])
        self.ops[2] = result

    def sharpen(self, a, temperature=0.05):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return a
