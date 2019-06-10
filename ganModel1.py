from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from PIL import Image
import os
from ops import imageProcess 

from GLOVEModel import GLOVEModel

class DCGan(object):
    modelName = 'dc-gan'
    
    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None

        self.imgWidth = 7
        self.imgHeight = 7
        self.imgChannels = 1
        self.randomInputDim = 50
        self.textInputDim = 100
        
        self.word2vecModel = GLOVEModel(self.textInputDim)

        self.config = None

    @staticmethod
    def getConfigFilePath(modelDirPath):
        return os.path.join(modelDirPath, DCGan.modelName + '-config.npy')

    @staticmethod
    def getWeightFilePath(modelDirPath, modelType):
        return os.path.join(modelDirPath, DCGan.modelName + '-' + modelType + '-weights.h5')

    def createGenerator(self,randomInput,textInput):
        initImgWidth = self.imgWidth // 16
        initImgHeight = self.imgHeight // 16

        randomOutput = Dense(256)(randomInput)
        textOutput = Dense(512)(textInput)

        randomOutput1 = Dense(512)(randomOutput)
        textOutput1 = Dense(1024)(textOutput)

        randomOutput2 = Dense(1024)(randomOutput1)
        textOutput2 = Dense(2048)(textOutput1)

        totalInput = concatenate([textOutput1,randomOutput1])
        layer1 = Activation('tanh')(totalInput)

        layer2 = Dense(128 * initImgWidth * initImgHeight)(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation('tanh')(layer2)
        layer2 = Reshape((initImgWidth,initImgHeight,128),input_shape = (128 * initImgWidth * initImgHeight,))(layer2)

        layer3 = UpSampling2D(size = (2,2))(layer2)
        layer3 = Conv2D(64,kernel_size = 5,padding='same')(layer3)
        layer3 = Activation('tanh')(layer3)

        layer4 = UpSampling2D(size = (2,2))(layer3)
        layer4 = Conv2D(32,kernel_size = 5,padding='same')(layer4)
        layer4 = Activation('tanh')(layer4)

        layer5 = UpSampling2D(size = (2,2))(layer4)
        layer5 = Conv2D(16,kernel_size = 5,padding='same')(layer5)
        layer5 = Activation('tanh')(layer5)

        layer6 = UpSampling2D(size = (2,2))(layer5)
        layer6 = Conv2D(self.imgChannels,kernel_size = 5,padding='same')(layer6)
        output = Activation('tanh')(layer6)

        self.generator = Model([randomInput,textInput],output)
        self.generator.compile(loss='mean_squared_error',optimizer='SGD')

        print('generator ', self.generator.summary())

    def createDiscrimminator(self,textInput,imgInput):
        textOutput = Dense(1024)(textInput)

        layer1 = Conv2D(64,kernel_size = 5,padding = 'same')(imgInput)
        layer1 = Activation('tanh')(layer1)

        layer2 = MaxPooling2D(pool_size = (2,2))(layer1)

        layer3 = Conv2D(128,kernel_size = 5)(layer2)
        layer3 = Activation('tanh')(layer3)

        layer4 = MaxPooling2D(pool_size = (2,2))(layer3)

        layer5 = Flatten()(layer4)
        imgOutput = Dense(1024)(layer5)

        totalInput = concatenate([imgOutput,textOutput])

        layer6 = Activation('tanh')(totalInput)
        layer6 = Dense(1)(layer6)

        output = Activation('sigmoid')(layer6)

        self.discriminator = Model([imgInput,textInput],output)
        d_optim = SGD(lr=0.0005, momentum=0.5, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        print('discriminator: ', self.discriminator.summary())

    def createModel(self):
        generatorTextInput = Input(shape = (self.textInputDim,))
        generatorRandomInput = Input(shape = (self.randomInputDim,))

        discriminatorTextInput = Input(shape = (self.textInputDim,))
        discriminatorImageInput = Input(shape = (self.imgWidth, self.imgHeight,self.imgChannels))

        self.createGenerator(generatorRandomInput,generatorTextInput)
        self.createDiscrimminator(discriminatorTextInput,discriminatorImageInput)
        
        output = self.discriminator([self.generator.output,generatorTextInput])

        self.model = Model([generatorRandomInput, generatorTextInput], output)
        self.discriminator.trainable = False

        g_optim = SGD(lr=0.0005, momentum=0.5, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        print('generator-discriminator: ', self.model.summary())

    def buildModel(self):
        self.createModel()

    def loadModel(self, modelDirPath):
        configFilePath = DCGan.getConfigFilePath(modelDirPath)
        self.config = np.load(configFilePath).item()
        self.imgWidth = self.config['imgWidth']
        self.imgHeight = self.config['imgHeight']
        self.imgChannels = self.config['imgChannels']
        self.randomInputDim = self.config['randomInputDim']
        self.textInputDim = self.config['textInputDim']

        self.buildModel()
        self.word2vecModel = GLOVEModel(self.textInputDim)
        self.generator.load_weights(DCGan.getWeightFilePath(modelDirPath,'generator'))
        self.discriminator.load_weights(DCGan.getWeightFilePath(modelDirPath, 'discriminator'))
        self.word2vecModel.build()

    def fit(self,modelDirPath,data,epochs = None, batchSize = None,snapshotDirPath = None, snapshotInterval = None):
        
        if (epochs == None):
            epochs = 100

        if (batchSize == None):
            batchSize = 128

        if (snapshotInterval == None):
            snapshotInterval = 20

        self.config = dict()
        self.config['imgWidth'] = self.imgWidth
        self.config['imgHeight'] = self.imgHeight
        self.config['randomInputDim'] = self.randomInputDim
        self.config['textInputDim'] = self.textInputDim
        self.config['imgChannels'] = self.imgChannels
        
        self.word2vecModel = GLOVEModel(self.textInputDim)
        self.word2vecModel.build()

        configFilePath = DCGan.getConfigFilePath(modelDirPath)

        np.save(configFilePath,self.config)
        
        randomBatch = np.zeros((batchSize,self.randomInputDim))
        textBatch = np.zeros((batchSize, self.textInputDim))

        self.buildModel()

        for epoch in range(epochs):
            batchCount = int(data.shape[0] / batchSize)

            for batchIndex in range(batchCount):
                dataBatch = data[batchIndex * batchSize:(batchIndex + 1) * batchSize]

                imageBatch = []
                for index in range(batchSize):
                    dataUnit = dataBatch[index]

                    img = dataUnit[0]
                    text = dataUnit[1]

                    imageBatch.append(img)

                    textBatch[index, :] = self.word2vecModel.encode(text)
                    randomBatch[index, :] = np.random.uniform(-1, 1, self.randomInputDim)

                imageBatch = np.array(imageBatch)

                generatedImages = self.generator.predict([randomBatch, textBatch], verbose=0)

                if (epoch * batchSize + batchIndex) % snapshotInterval == 0:
                    self.saveSnapshots(generatedImages,snapshotDirPath,epoch,batchIndex)

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((imageBatch, generatedImages)),
                                                            np.concatenate((textBatch, textBatch))],
                                                           np.array([1] * batchSize + [0] * batchSize))

                print("Epoch %d batch %d d_loss : %f" % (epoch, batchIndex, d_loss))

                for index in range(batchSize):
                    randomBatch[index, :] = np.random.uniform(-1, 1, self.randomInputDim)

                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch([randomBatch, textBatch], np.array([1] * batchSize))

                print("Epoch %d batch %d g_loss : %f" % (epoch, batchIndex, g_loss))

        self.generator.save_weights(DCGan.getWeightFilePath(modelDirPath, 'generator'), True)
        self.discriminator.save_weights(DCGan.getWeightFilePath(modelDirPath, 'discriminator'), True)

    def generateImage(self, text):
        randomInput = np.zeros(shape=(1, self.randomInputDim))
        textInput = np.zeros(shape=(1, self.textInputDim))

        randomInput[0, :] = np.random.uniform(-1, 1, self.randomInputDim)
        textInput[0, :] = self.word2vecModel.encode(text)

        image = self.generator.predict([randomInput, textInput], verbose=0)

        image = image[0]
        image = image * 127.5 + 127.5

        return Image.fromarray(image.astype(np.uint8))

    def saveSnapshots(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = imageProcess.imageMerge(generated_images)

        imageProcess.UndoNormalizedImg(image).save(
            os.path.join(snapshot_dir_path, DCGan.modelName + '-' + str(epoch) + "-" + str(batch_index) + ".png"))