import numpy as np
import os
import re

class GLOVEModel(object):
    def __init__(self,dim):
        self.vecDim = dim
        self.gloveFolderPath = 'data/glove/'
        self.gloveFilePath = ''
        self.dictionary = {}

    def build(self):
        self.gloveFilePath = self.gloveFolderPath + 'glove.6B.' + str(self.vecDim) + 'd.txt'
        file = open(self.gloveFilePath,mode='rt',encoding = 'utf-8')

        for line in file:
            words = line.strip().split()
            self.dictionary[words[0]] = np.array(words[1:],dtype=np.float32)
        file.close()

    def encode(self,sentence):
        words = []
        for word in re.split('[,.?!: \n]',sentence):
            words.append(word.lower())

        length = len(words)
        vec = np.zeros(shape=(self.vecDim,))

        for j in range(length):
            word = words[j]
            if (word != ''):
                try:
                    vec[:] = vec[:] + self.dictionary[word][:]
                except KeyError:
                    pass
                
        return vec

def test():
    model = GLOVEModel(50)
    model.build()
    str = "this is a dog"
    vec = model.encode(str)
    print(vec)
    print(len(vec))