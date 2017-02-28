import numpy as np
result = np.load("iclr_result.npy")
memory= [x[0] for x in result]
registers = [x[1] for x in result]
print memory[0]
print "*****"
print registers[0]
print "****"
print memory[-1]
print "*****"
print registers[-1]
#print registers[100]
