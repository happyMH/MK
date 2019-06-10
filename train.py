import os 
import sys 
import numpy as np
from random import shuffle
from ops import readData

from ganModel1 import DCGan
from ops import readData


def train():
    seed = 0
    np.random.seed(seed)
    
    current_dir = os.path.dirname(__file__)
    # add the keras_text_to_image module to the system path
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    img_dir_path = 'data/flower/img'
    txt_dir_path = 'data/flower/txt/text_c10'
    model_dir_path = 'data/models'

    img_width = 64
    img_height = 64
    img_channels = 3

    data = readData.readData(img_dir_path, txt_dir_path, img_width, img_height)

    shuffle(data)

    gan = DCGan()
    gan.imgWidth = img_width
    gan.imgHeight = img_height
    gan.imgChannels = img_channels
    gan.randomInputDim = 100

    batch_size = 9
    epochs = 5000
    gan.fit(modelDirPath=model_dir_path, data=data,
            snapshotDirPath='data/flower/snapshots',
            snapshotInterval=100,
            batchSize=batch_size,
            epochs=epochs)

if __name__ == '__main__':
    train()

