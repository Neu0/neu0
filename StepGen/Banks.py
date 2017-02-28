import numpy as np

import math


class RegisterBank:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.memory = [0] * size

    def read_fuzzy(self, weights):
        '''
        :param weights: numpy array with weights for each cell
        :return: value read fuzzily
        '''
        weights = weights.tolist()
        value = 0
        for weight, cell_value in zip(weights, self.memory):
            value += weight * cell_value
        return value

    def read(self, index):
        return self.memory[index]

    def write(self, index, value):
        self.memory[index] = value

    def get_memory(self):
        return self.memory

    def write_fuzzy(self, weights, value):
        """
        :param weights:  numpy array with weights for each cell
        :param value: value to write to each cell according to weights
        :return: None
        """
        weights = weights.tolist()
        # retain
        retain_weights = [1 - x for x in weights]
        # We do not need erase vector
        for index in range(len(self.memory)):
            self.memory[index] = retain_weights[index] * self.memory[index]
        # Add
        for index in range(len(self.memory)):
            self.memory[index] += weights[index] * value


class MemoryBank:
    def __init__(self, name, size, variance=0.05):
        self.size = size
        self.name = name
        self.memory = [0] * size
        self.variance = variance

    def make_gaussian(self, mean):
        scale = 1 / math.sqrt(2 * math.pi * self.variance)
        return lambda x: scale * math.e ** -((x - mean) ** 2 / (2 * self.variance))

    def get_threshold(self, gauss, value):
        lower = int(value)
        while (gauss(lower) > 0):
            lower -= 1
        return int(value) - lower

    def read(self,index):
        return self.memory[index]

    def write(self,index,value):
        self.memory[index] = value

    def read_gauss(self, value):
        gauss = self.make_gaussian(value)
        #thresh = self.get_threshold(gauss, value)
        #range_ = (int(value) - thresh, int(value) + thresh)
        #range_ = (max(range_[0], 0), min(range_[1], self.size))
        range_ = (0,self.size)
        probs = {}  # * self.size
        sum_ = 0
        for mem in range(range_[0], range_[1]):
            probs[mem] = math.e ** gauss(mem)
            sum_ += probs[mem]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_
        sum_ = 0
        for x in range(range_[0], range_[1]):
            probs[x] = math.e ** (math.log(probs[x]) / 0.05)
            sum_ += probs[x]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_

        value = 0
        for mem in range(range_[0], range_[1]):
            value += self.memory[mem] * probs[mem]
        return value

    def write_gauss(self, value, content):
        gauss = self.make_gaussian(value)
        #thresh = self.get_threshold(gauss, value)
        #range_ = (int(value) - thresh, int(value) + thresh)
        #range_ = (max(range_[0], 0), min(range_[1], self.size))
        range_ = (0,self.size)
        probs = {}
        sum_ = 0
        for mem in range(range_[0], range_[1]):
            probs[mem] = math.e ** gauss(mem)
            sum_ += probs[mem]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_
        sum_ = 0
        for x in range(range_[0], range_[1]):
            probs[x] = math.e ** (math.log(probs[x]) / 0.05)
            sum_ += probs[x]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_
        for mem in range(range_[0], range_[1]):
            self.memory[mem] = (1 - probs[mem]) * self.memory[mem]
        for mem in range(range_[0], range_[1]):
            self.memory[mem] += probs[mem] * content


    def get_memory(self):
        return self.memory


class CacheBank:
    def __init__(self,size=200,memory_size = 100000):
        self.size = size
        self.locations = [-1]*size
        self.contents = [0]*size
        self.decay = [0]*size
        self.beta = np.load("betasquare.npy")
        self.w1 = np.load("w1square_200.npy")
        self.memory = MemoryBank("MAIN",memory_size)

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference
    def disable(self):
        self.temp = self.read
        self.temp2 = self.write
        self.read = self.memory.read_gauss
        self.write = self.memory.write_gauss

    def enable(self):
        self.read = self.temp
        self.write = self.temp2

    def get_probs(self,value):
        sample = np.array([(x-value)**2 for x in self.locations])
        hidden_act = self.beta * sample
        hidden_act = np.tanh(hidden_act)
        soft_in = self.w1.dot(hidden_act)
        soft_out = self.softmax(soft_in)
        return soft_out

    def read(self,value):
        probs = self.get_probs(value)
        if(np.argmax(probs) == self.size):
            content = self.memory.read_gauss(value)
            location = self.decay.index(max(self.decay))
            if self.locations[location]!=-1:
                self.memory.write_gauss(self.locations[location],self.contents[location])
            self.locations[location] = value
            self.contents[location] = content
            self.decay = [1+x for x in self.decay]
            self.decay[location] = 0
        else:
            probs = self.sharpen(probs)
            content = 0
            self.decay = [1+x for x in self.decay]
            for i in range(self.size):
                content += self.contents[i] * probs[i]
                self.decay[i] = (1-probs[i]) * self.decay[i]
        return content

    def write(self,value,content):
        #Value is location
        probs = self.get_probs(value)
        if (np.argmax(probs) == self.size):
            location = self.decay.index(max(self.decay))
            if self.locations[location]!=-1:
                self.memory.write_gauss(self.locations[location],self.contents[location])
            self.locations[location] = value
            self.contents[location] = content
            self.decay = [1 + x for x in self.decay]
            self.decay[location] = 0
        else:
            probs = self.sharpen(probs)
            # retain
            retain_weights = [1 - x for x in probs]
            self.decay = [1 + x for x in self.decay]
            # We do not need erase vector
            for index in range(self.size):
                self.contents[index] = retain_weights[index] * self.contents[index]
            # Add
            for index in range(self.size):
                self.contents[index] += probs[index] * content
                self.decay[index] = (1-probs[index]) * self.decay[index]

    def flush_all(self):
        for location,content in enumerate(self.locations,self.contents):
            if(location!=-1):
                self.memory.write_gauss(location,content)


    def sharpen(self, a, temperature=0.05):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return a

    def get_memory(self):
        print self.locations
        print self.contents
        print self.decay
