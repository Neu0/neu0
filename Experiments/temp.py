def sharpen(a, temperature=0.05):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return a

l=[0.1,0.2,0.7]
import numpy as np
print sharpen(np.array(l))