class MIPS:
    def __init__(self):
        self.load = []
        self.load_targets = []
        self.store = []
        self.add = []
        self.subtract = []
        self.multiply = []
        self.division = []
        self.branch = []
        self.branch_condition = []
        self.instructions = ["add","sub","b","mul","mov","cmp","str","ldr"]
        self.conds = ["", "ne", "eq", "gt", "lt"]
        self.num_regs = 15
        self.num_mips = 8
        self.total =[]
        self.makeLoadInstruction()
        self.makeArithmetic()
        self.makeBranch()
        self.makeStoreInstruction()
        self.total = self.load+self.add+self.subtract+self.branch+self.branch_condition+self.store
        self.total = self.total * 20

    def makeLoadInstruction(self):

        inst = "la"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                instruction = inst + " " + "$t" + str(r0) + ",$t"+str(r1)
                r0_op = [0] * self.num_regs
                r0_op[r0] = 1
                r2_op = [0] * self.num_regs
                r2_op[r1] = 1

                r1_op = [0] * self.num_regs
                r1_op[self.num_regs - 1] = 1
                i = [0] * len(self.instructions)
                i[self.instructions.index("mov")] = 1
                cond = [0] * len(self.conds)
                cond[0] = 1
                self.load.append((instruction, (i, r0_op, r2_op, r1_op, cond, [1, 0])))

        inst = "lw"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                instruction = inst + " " + "$t" + str(r0) + ",($t"+str(r1) + ")"
                r0_op = [0] * self.num_regs
                r0_op[r0] = 1
                r2_op = [0] * self.num_regs
                r2_op[r1] = 1

                r1_op = [0] * self.num_regs
                r1_op[self.num_regs - 1] = 1
                i = [0] * len(self.instructions)
                i[self.instructions.index("mov")] = 1
                cond = [0] * len(self.conds)
                cond[0] = 1
                self.load.append((instruction, (i, r0_op, r2_op, r1_op, cond, [0, 1])))

    def makeStoreInstruction(self):

        inst = "sw"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                instruction = inst + " " + "$t" + str(r0) + ",($t"+ str(r1) + ")"
                r0_op = [0] * self.num_regs
                r0_op[r0] = 1
                r2_op = [0] * self.num_regs
                r2_op[r1] = 1

                r1_op = [0] * self.num_regs
                r1_op[self.num_regs - 1] = 1
                i = [0] * len(self.instructions)
                i[self.instructions.index("str")] = 1
                cond = [0] * len(self.conds)
                cond[0] = 1
                self.store.append((instruction, (i, r0_op, r1_op, r2_op, cond, [0, 1])))

    def makeArithmetic(self):
        inst = "add"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                for r2 in range(0, self.num_mips):
                    instruction = inst + " " + "$t" + str(r0)+ ",$t" + str(r1) + ",$t" + str(r2)
                    r0_op = [0] * self.num_regs
                    r0_op[r0] = 1
                    r1_op = [0] * self.num_regs
                    r1_op[r1] = 1
                    r2_op = [0] * self.num_regs
                    r2_op[r2] = 1
                    i = [0] * len(self.instructions)
                    i[self.instructions.index(inst)] = 1
                    cond = [0] * len(self.conds)
                    cond[0] = 1
                    self.add.append((instruction, (i, r0_op, r1_op, r2_op, cond, [1, 0])))

        inst = "sub"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                for r2 in range(0, self.num_mips):
                    instruction = inst + " " + "$t" + str(r0)+",$t" + str(r1) + ",$t" + str(r2)
                    r0_op = [0] * self.num_regs
                    r0_op[r0] = 1
                    r1_op = [0] * self.num_regs
                    r1_op[r1] = 1
                    r2_op = [0] * self.num_regs
                    r2_op[r2] = 1
                    i = [0] * len(self.instructions)
                    i[self.instructions.index(inst)] = 1
                    cond = [0] * len(self.conds)
                    cond[0] = 1
                    self.add.append((instruction, (i, r0_op, r1_op, r2_op, cond, [1, 0])))

        '''
        multiply self.num_regs-bit quantities in $t3 and $t4, and store 64-bit
		result in special registers Lo and Hi:  (Hi,Lo) = $t3 * $t4
        '''
        '''
        inst = "mult"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                instruction = inst + " " + "$t" + str(r0) + ",$t" + str(r1)
                r0_op = [0] * self.num_regs
                r0_op[r0 + self.num_mips] = 1
                r1_op = [0] * self.num_regs
                r1_op[r1] = 1
                r2_op = [0] * self.num_regs
                r2_op[self.num_regs - 1] = 1
                i = [0] * len(self.instructions)
                i[self.instructions.index(inst)] = 1
                cond = [0] * len(self.conds)
                cond[0] = 1
                self.multiply.append((instruction, (i, r0_op, r1_op, r2_op, cond, [1, 0])))
        '''

    def makeDivision(self):
        # Lo = $t5 / $t6   (integer quotient)
        #  Hi = $t5 mod $t6   (remainder)
        inst = "div"
        for r0 in range(0, self.num_mips):
            for r1 in range(0, self.num_mips):
                instruction = inst + " " + "$t" + str(r0) + ",$t" + str(r1)
                r0_op = [0] * self.num_regs
                r0_op[r0 + self.num_mips] = 1
                r1_op = [0] * self.num_regs
                r1_op[r1] = 1
                r2_op = [0] * self.num_regs
                r2_op[self.num_regs - 1] = 1
                i = [0] * len(self.instructions)
                i[self.instructions.index(inst)] = 1
                cond = [0] * len(self.conds)
                cond[0] = 1
                self.division.append((instruction, (i, r0_op, r1_op, r2_op, cond, [1, 0])))

    def makeBranch(self):
        inst = "b"
        for r0 in range(0, self.num_mips):
            instruction = inst + " " + "$t" + str(r0)
            r0_op = [0] * self.num_regs
            r0_op[r0] = 1
            r1_op = [0] * self.num_regs
            r1_op[self.num_regs - 1] = 1
            r2_op = [0] * self.num_regs
            r2_op[self.num_regs - 1] = 1
            i = [0] * len(self.instructions)
            i[self.instructions.index(inst)] = 1
            cond = [0] * len(self.conds)
            cond[0] = 1
            self.branch.append((instruction, (i, r2_op, r2_op, r0_op, cond, [1, 0])))

        for condition in self.conds:
            for r0 in range(0, self.num_mips):
                for r1 in range(0, self.num_mips):
                    for r2 in range(0, self.num_mips):
                        instruction = "b" + condition + " " + "$t" + str(r0)+ ",$t" + str(r1) + ",$t" + str(r2)
                        r0_op = [0] * self.num_regs
                        r0_op[r0] = 1
                        r1_op = [0] * self.num_regs
                        r1_op[r1] = 1
                        r2_op = [0] * self.num_regs
                        r2_op[r2] = 1
                        i = [0] * len(self.instructions)
                        i[self.instructions.index(inst)] = 1
                        cond = [0] * len(self.conds)
                        cond[self.conds.index(condition)] = 1
                        self.branch_condition.append((instruction, (i, r0_op, r1_op, r2_op, cond, [1, 0])))
