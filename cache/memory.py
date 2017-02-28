from __future__ import division
import numpy as np
import random
cache_size = 1000
l = [i for i in range(cache_size)]
samples = []
targets= []

def diff_fun(a,b):
    return (a-b)**2
before = []
for ind in range(1000):
    contents = l

    key_in = ind
    noise = random.randint(0,10)/100
    sign = 1 if random.random() > 0.5 else -1
    #key_in = random.sample(contents,1)[0]
    target1 = [0]*(cache_size+1)
    target1[contents.index(key_in)] = 1
    key_in += noise * sign
    #samples.append(np.array([diff_fun(x,key_in) for x in contents]))
    #targets.append(np.array(target1))
    before.append([np.array([diff_fun(x,key_in) for x in contents]),np.array(target1)])
import random
random.shuffle(before)
samples = [np.tanh(x[0]) for x in before[:10]]
targets = [np.tanh(x[1]) for x in before[:10]]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

inp_size = cache_size
beta = np.array([1.0])
w1 = np.random.randn(cache_size+1,cache_size)
mem = np.zeros_like(w1)
membeta= np.zeros_like(beta)

def forward(soft_in):
    #hidden_act = beta * sample
    #hidden_act = np.tanh(hidden_act)
    #soft_in = w1.dot(hidden_act)
    soft_out = sharpen(softmax(-1000*soft_in))
    return soft_out

def sharpen( a, temperature=0.05):
    return a
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return a
def backward(error,hidden_act,sample):

    error = error.reshape((cache_size + 1, 1))
    dw1 = error.dot(hidden_act.reshape((1, cache_size)))
    return dw1


learning_rate = 0.005

for epoch in range(000):
    loss = 0
    for sample,target in zip(samples,targets):
        prediction,hidden_act = forward(sample)
        loss += - np.sum(target * np.log(prediction))
        error = target - prediction
        #dw1 = backward(error,hidden_act,sample)
        #np.clip(dw1, -5, 5, out=dw1)
        #mem += dw1 * dw1
        #w1 += learning_rate * dw1 / np.sqrt(mem + 1e-8)  # adagrad update

        #w1 += learning_rate * dw1
        #beta += learning_rate * dbeta
    print(epoch,loss,beta)

#np.save("mem_w1.npy",w1)
#np.save("mem_beta.npy",beta)
#w1 = np.load("mem_w1.npy")

key = 500.1
a = (np.array([diff_fun(x,key) for x in l]))
print a
v = forward(a)
s=list( set(v.tolist()))
print len(s)


print np.argmax(v)