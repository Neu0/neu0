from __future__ import division
import numpy as np
import random
cache_size = 200
l = [i for i in range(500)]
samples = []
targets= []

def diff_fun(a,b):
    return (a-b)**2

for ind in range(10000):
    contents = random.sample(l,cache_size+1)
    contents,key_out = contents[:-1],contents[-1]
    noise = random.randint(0,10)/100
    sign = 1 if random.random() > 0.5 else -1
    key_in = random.sample(contents,1)[0]
    target1 = [0]*(cache_size+1)
    target2 = [0]*(cache_size+1)
    target1[contents.index(key_in)] = 1
    target2[-1] = 1
    key_in += noise * sign
    samples.append(np.array([diff_fun(x,key_in) for x in contents]))
    samples.append(np.array([diff_fun(x,key_out) for x in contents]))
    targets.append(np.array(target1))
    targets.append(np.array(target2))
print samples[10]
print targets[10]
print samples[11]
print targets[11]


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


inp_size = cache_size
beta = np.array([1.0])
w1 = np.random.randn(cache_size+1,cache_size)



def forward(sample):
    hidden_act = beta * sample
    hidden_act = np.tanh(hidden_act)
    soft_in = w1.dot(hidden_act)
    soft_out = softmax(soft_in)
    return soft_out,hidden_act

def backward(error,hidden_act,sample):
    error = error.reshape((cache_size + 1, 1))
    dw1 = error.dot(hidden_act.reshape((1, cache_size)))
    dhidden = w1.T.dot(error)
    d_lower = ((1-hidden_act**2).reshape(cache_size,1))*dhidden
    dbeta = d_lower.T.dot(sample)
    return dw1,dbeta


learning_rate = 0.01

for epoch in range(100):
    loss = 0
    for sample,target in zip(samples,targets):
        prediction,hidden_act = forward(sample)
        loss += - np.sum(target * np.log(prediction))
        error = target - prediction
        dw1,dbeta = backward(error,hidden_act,sample)
        w1 += learning_rate * dw1
        beta += learning_rate * dbeta
    print(epoch,loss,beta)

np.save("w1square_200.npy",w1)
np.save("betasquare.npy",beta)
f = [i for i in range(1000)]
contents = random.sample(f,cache_size+1)
contents,key_out = contents[:-1],contents[-1]
key_in = random.sample(contents,1)[0]+1-0.1
sample1 = np.array([diff_fun(x,key_in) for x in contents])
sample2 = np.array([diff_fun(x,key_out) for x in contents])
print contents,key_in,key_out
res = forward(sample1)
argmax = np.argmax(res[0])
print argmax
#print forward(sample2)