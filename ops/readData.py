import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

def readData(imgDirPath,textDirPath,imgWidth,imgHeight):
    data = []
    count = 0

    for root, dirs, files in os.walk(textDirPath):
        for f in files:
            if (f.endswith('txt')):
                count +=1
                print(count)
                if (count == 16):
                    return np.array(data)
                imgPath = os.path.join(imgDirPath,f[:-4]+'.jpg')
                image = load_img(imgPath, target_size=(imgWidth, imgHeight))
                imageArray = img_to_array(image)
                imageArray = (imageArray.astype(np.float32) / 255) * 2 -1

                filePath = os.path.join(root,f)
                file = open(filePath,mode='rt')
                line = file.readline()
                #while(line):
                data.append([imageArray,line])
                #line = file.readline()

    return np.array(data)


def test():
    imgDir = 'data/flower/img'
    txtDir = 'data/flower/txt'

    width = 64
    height = 64

    result = readData(imgDir,txtDir,width,height)
    print(result)
    

            


