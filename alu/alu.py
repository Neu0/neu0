import numpy as np


class ALU:
    def __init__(self):
        self.arith = np.load("mult.npy")
        temp = np.load("aluweights.npy")
        self.add = temp[0]
        self.sub = temp[1]
        self.mul = temp[2]

    def compute(self, inst, op1, op2, bias=1):
        inst0 = inst[0]
        inst1 = inst[1]
        inst2 = inst[2]
        input_alu = np.array([inst0, inst1, inst2, op1, op2, bias]).reshape(1, 6)
        add = self.perform_one_arith(self.add, input_alu)
        sub = self.perform_one_arith(self.sub, input_alu)
        mul = self.perform_one_arith(self.mul, input_alu)
        return add + sub + mul

    def perform_one_arith(self, weight, inp):
        out = inp.dot(weight)[0]
        out_op = self.add_sub_multiply(out[0], out[1], out[2])
        confidence = out[3]
        return out_op * confidence

    def add_sub_multiply(self, op1, op2, op3, bias=1):
        while True:
            input = np.array([op1, op2, op3, bias])
            res = input.reshape(1, 4).dot(self.arith)[0]
            op1 = res[0]
            op2 = res[1]
            op3 = res[2]
            if round(op2) == 0:
                return op1


if __name__ == "__main__":
    alu = ALU()
    while (True):
        try:
            add_prob = float(raw_input("Enter addition probability: "))
            sub_prob = float(raw_input("Enter subtraction probability: "))
            mul_prob = float(raw_input("Enter multiplication probability: "))
            op1 = float(raw_input("Enter op1: "))
            op2 = float(raw_input("Enter op2: "))
            print alu.compute([add_prob, sub_prob, mul_prob], op1, op2)
        except:
            print "Exception,Try again"
