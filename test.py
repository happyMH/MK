import os 
import sys 
import numpy as np
from random import shuffle
from ops.imageProcess import imageMerge
from ops.imageProcess import UndoNormalizedImg
from ops.readData import readData
from ganModel1 import DCGan


def test():    
    imgDirPath = 'data/flower/img'
    textDirPath = 'data/flower/txt'
    modelDirPath = 'data/models/flower'

    imgWidth = 32
    imgHeight = 32

    data = readData(imgDirPath, textDirPath, imgWidth, imgHeight)

    #shuffle(data)

    gan = DCGan()
    gan.loadModel(modelDirPath)

    pictureNum = 3
    repeatNum = 1

    for i in range(pictureNum):
        dataUnit = data[i]
        img = dataUnit[0]
        text = dataUnit[1]

        for j in range(repeatNum):
            result = gan.generateImage(text)
            result.save('data/flower/test/' + DCGan.modelName + '-generated-' + str(i) + '-' + str(j) + '.jpg')


if __name__ == '__main__':
    test()