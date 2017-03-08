class ARM:
    def __init__(self,num_regs):
        self.load = []
        self.load_targets = []
        self.store = []
        self.add = []
        self.compare =[]
        self.subtract = []
        self.multiply = []
        self.division = []
        self.move = []
        self.branch = []
        self.branch_condition = []
        self.instructions = ["add","sub","b","mul","mov","cmp","str","ldr"]
        self.conds = ["", "ne", "eq", "gt", "lt"]
        self.num_regs = num_regs
        self.no_val = -1

        self.total =[]
        self.makeLoadInstruction()
        self.makeArithmetic()
        self.makeBranch()
        self.makeStoreInstruction()
        self.makeCompare()
        self.makeMove()
        self.total = self.load+self.add+self.branch+self.compare+self.store+self.move

    def makeLoadInstruction(self):
        op = "ldr"
        for r0 in range(self.num_regs - 1):  # 14
            for cond in self.conds:  # 5
                for j in range(15):  # 15
                    for r2 in range(0, self.num_regs - 1):  # 14
                        inst = op + cond + " R" + str(r0) + "," + "[R" + str(r2) + "]"
                        i_r2 = [0] * self.num_regs
                        i_r2[r2] = 1
                        i_r1 = [0] * self.num_regs
                        i_r1[-1] = 1
                        i_r0 = [0] * self.num_regs
                        i_r0[r0] = 1
                        i = [0] * len(self.instructions)
                        i[self.instructions.index(op)] = 1
                        i_cond = [0] * len(self.conds)
                        i_cond[self.conds.index(cond)] = 1
                        self.load.append((inst, [i, i_r0, i_r2, i_r1, i_cond, [0, 1],self.no_val]))


    def makeStoreInstruction(self):

        op = "str"
        for r0 in range(self.num_regs - 1):  # 14
            for cond in self.conds:  # 5
                for j in range(15):  # 15
                    for r2 in range(0, self.num_regs - 1):  # 14
                        inst = op + cond + " R" + str(r0) + "," + "[R" + str(r2) + "]"
                        i_r2 = [0] * self.num_regs
                        i_r2[r2] = 1
                        i_r1 = [0] * self.num_regs
                        i_r1[-1] = 1
                        i_r0 = [0] * self.num_regs
                        i_r0[r0] = 1
                        i = [0] * len(self.instructions)
                        i[self.instructions.index(op)] = 1
                        i_cond = [0] * len(self.conds)
                        i_cond[self.conds.index(cond)] = 1
                        self.store.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [0, 1],self.no_val]))

    def makeArithmetic(self):
        ops=["add","sub","mul"]
        i_bracket = [1, 0]
        for op in ops:
            for cond in self.conds:  # 5
                for r0 in range(0, self.num_regs - 1):  # 14
                    for r1 in range(0, self.num_regs - 1):  # 14
                        for r2 in range(0, self.num_regs - 1):  # 14 : 27,440
                            if (r0 == r1 or r0 == r2):
                                continue
                            inst = op + cond + " R" + str(r0) + "," +  "R" + str(r1) + ",R" + str(r2)
                            i_r2 = [0] * self.num_regs
                            i_r2[r2] = 1
                            # samples.append(inst)
                            i = [0] * len(self.instructions)
                            i[self.instructions.index(op)] = 1
                            i_cond = [0] * len(self.conds)
                            i_cond[self.conds.index(cond)] = 1
                            i_r0 = [0] * self.num_regs
                            i_r0[r0] = 1
                            i_r1 = [0] * self.num_regs
                            i_r1[r1] = 1
                            self.add.append((inst, [i, i_r0, i_r1, i_r2, i_cond, i_bracket,self.no_val]))

                        for const in range(0, 50):  # 14 : 27,440
                            if (r0 == r1):
                                continue
                            inst = op + cond + " R" + str(r0) + "," +  "R" + str(r1) + ",#" + str(const)
                            i_r2 = [0] * self.num_regs
                            i_r2[self.num_regs - 1 ] = 1
                            # samples.append(inst)
                            i = [0] * len(self.instructions)
                            i[self.instructions.index(op)] = 1
                            i_cond = [0] * len(self.conds)
                            i_cond[self.conds.index(cond)] = 1
                            i_r0 = [0] * self.num_regs
                            i_r0[r0] = 1
                            i_r1 = [0] * self.num_regs
                            i_r1[r1] = 1
                            self.add.append((inst, [i, i_r0, i_r1, i_r2, i_cond, i_bracket,const]))

    def makeBranch(self):
        for j in range(50):
            for cond in self.conds:  # 5
                for const in range(50):  # 14
                    inst = "B" + cond + " #" + str(const)
                    r_out = [0] * self.num_regs
                    r_out[-1] = 1
                    r2_out = [0] * self.num_regs
                    r2_out[self.num_regs -1] = 1
                    i_inst = [0] * len(self.instructions)
                    i_inst[self.instructions.index("b")] = 1
                    i_cond = [0] * len(self.conds)
                    i_cond[self.conds.index(cond)] = 1
                    self.branch.append((inst, [i_inst, r_out, r_out, r2_out, i_cond, [1, 0],const]))

    def makeCompare(self):
        op = "cmp"
        for j in range(1):
            regs = self.num_regs
            for r0 in range(regs - 1):
                for k in range(50):
                    for r2 in range(0, regs - 1):
                        inst = op + " R" + str(r0) + "," + "R" + str(r2)
                        i_r2 = [0] * regs
                        i_r2[r2] = 1
                        i_r1 = [0] * regs
                        i_r1[-1] = 1
                        i_r0 = [0] * regs
                        i_r0[r0] = 1
                        i = [0] * len(self.instructions)
                        i[self.instructions.index(op)] = 1
                        i_cond = [0] * len(self.conds)
                        i_cond[self.conds.index("")] = 1
                        self.compare.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [1, 0],self.no_val]))

                for const in range(0, 50):
                    inst = op + " R" + str(r0) + "," + "#" + str(const)
                    i_r2 = [0] * regs
                    i_r2[self.num_regs-1] = 1
                    i_r1 = [0] * regs
                    i_r1[-1] = 1
                    i_r0 = [0] * regs
                    i_r0[r0] = 1
                    i = [0] * len(self.instructions)
                    i[self.instructions.index(op)] = 1
                    i_cond = [0] * len(self.conds)
                    i_cond[self.conds.index("")] = 1
                    self.compare.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [1, 0],const]))
    def makeMove(self):
        op = "mov"

        for r0 in range(self.num_regs - 1):  # 14
            for cond in self.conds:  # 5
                for j in range(5):  # 15
                    for r2 in range(0, self.num_regs - 1):  # 14
                        inst = op + cond + " R" + str(r0) + "," + "R" + str(r2)
                        i_r2 = [0] * self.num_regs
                        i_r2[r2] = 1
                        i_r1 = [0] * self.num_regs
                        i_r1[-1] = 1
                        i_r0 = [0] * self.num_regs
                        i_r0[r0] = 1
                        i = [0] * len(self.instructions)
                        i[self.instructions.index(op)] = 1
                        i_cond = [0] * len(self.conds)
                        i_cond[self.conds.index(cond)] = 1
                        self.move.append((inst, [i, i_r0, i_r2, i_r1, i_cond, [1, 0],self.no_val]))

                        for r2 in range(0, self.num_regs - 1):  # 14
                            inst = op + cond + " PC" + "," + "R" + str(r2)
                            i_r2 = [0] * self.num_regs
                            i_r2[r2] = 1
                            i_r1 = [0] * self.num_regs
                            i_r1[-1] = 1
                            i_r0 = [0] * self.num_regs
                            i_r0[-2] = 1
                            i = [0] * len(self.instructions)
                            i[self.instructions.index(op)] = 1
                            i_cond = [0] * len(self.conds)
                            i_cond[self.conds.index(cond)] = 1
                            self.move.append((inst, [i, i_r0, i_r2, i_r1, i_cond, [1, 0], self.no_val]))

                        for r2 in range(0, self.num_regs - 1):  # 14
                            inst = op + cond + " R" + str(r2) + "," + "PC"
                            i_r2 = [0] * self.num_regs
                            i_r2[-2] = 1
                            i_r1 = [0] * self.num_regs
                            i_r1[-1] = 1
                            i_r0 = [0] * self.num_regs
                            i_r0[r2] = 1
                            i = [0] * len(self.instructions)
                            i[self.instructions.index(op)] = 1
                            i_cond = [0] * len(self.conds)
                            i_cond[self.conds.index(cond)] = 1
                            self.move.append((inst, [i, i_r0, i_r2, i_r1, i_cond, [1, 0], self.no_val]))
                for const in range(0, 50):  # 14
                    inst = op + cond + " R" + str(r0) + "," + "#" + str(const)
                    i_r2 = [0] * self.num_regs
                    i_r2[self.num_regs -1] = 1
                    i_r1 = [0] * self.num_regs
                    i_r1[-1] = 1
                    i_r0 = [0] * self.num_regs
                    i_r0[r0] = 1
                    i = [0] * len(self.instructions)
                    i[self.instructions.index(op)] = 1
                    i_cond = [0] * len(self.conds)
                    i_cond[self.conds.index(cond)] = 1
                    self.move.append((inst, [i, i_r0, i_r2, i_r1, i_cond, [1, 0],const]))



