import argparse
import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
#import tensorflow.compat.v1 as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import random
import numpy as np
import scipy.io
from model import lee_mdoel,Cov1_model,RNN_model
import tensorflow.python.keras.saving.saved_model.model_serialization
from scipy.ndimage.filters import gaussian_filter1d
import scipy.special
import pickle
from dataPreprocessing import prepareDataCubesForRNN
import sys
from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# from keras.callbacks import TensorBoard

tf.compat.v1.disable_eager_execution()
#np.set_printoptions(threshold=sys.maxsize)
tf.compat.v1.experimental.output_all_intermediates(True)
class charSeqRNN(object):
    """
    This class encapsulates all the functionality needed for training, loading and running the handwriting decoder RNN.
    To use it, initialize this class and then call .train() or .inference(). It can also be run from the command line (see bottom
    of the script). The args dictionary passed during initialization is used to configure all aspects of its behavior.
    """

    def __init__(self, args):
        """
        This function initializes the entire tensorflow graph, including the dataset pipeline and RNN.
        Along the way, it loads all relevant data and label files needed for training, and initializes the RNN variables to
        default values (or loads them from a specified file). After initialization is complete, we are ready
        to either train (charSeqRNN.train) or infer (charSeqRNN.inference).
        """
        self.args = args

        # parse whether we are loading a model or not, and whether we are training or 'running' (inferring)
        if self.args['mode'] == 'train':
            self.isTraining = True
            ckpt = tf.train.get_checkpoint_state(self.args['loadDir'])
            if ckpt == None:
                # Nothing to load (no checkpoint found here), so we won't resume or try to load anything
                self.loadingInitParams = False
                self.resumeTraining = False
            elif self.args['loadDir'] == self.args['outputDir']:
                # loading from the same place we are saving - assume we are resuming training
                self.loadingInitParams = True
                self.resumeTraining = True
            else:
                # otherwise we will load params but not try to resume a training run, we'll start over
                self.loadingInitParams = True
                self.resumeTraining = False

        elif self.args['mode'] == 'infer':
            self.isTraining = False
            self.loadingInitParams = True
            self.resumeTraining = False

        # count how many days of data are specified
        self.nDays = 0
        for t in range(30):
            if 'labelsFile_' + str(t) not in self.args.keys():
                self.nDays = t
                break

        # load data, labels, train/test partitions & synthetic .tfrecord files for all days
        neuralCube_all, targets_all, errWeights_all, numBinsPerTrial_all, cvIdx_all, recordFileSet_all = self._loadAllDatasets()

        # index=1
        # neuralCube_all0=neuralCube_all[index][:,:,:]
        # targets_all0 = targets_all[index][:, :, 0:-1]
        #print(neuralCube_all0.shape)

        # for i in range(0,9):
        #     train_ALL10=np.concatenate((neuralCube_all[i],neuralCube_all[i+1]),axis=0)
        #     test_ALL10=np.concatenate((targets_all[i],targets_all[i+1]),axis=0)

        # print('begining')
        # f1=open('neuralCube.txt',mode='w',encoding='utf-8')
        # f2=open('target.txt',mode='w',encoding='utf-8')
        # f1.writelines(str(neuralCube_all0[:100,:,:]))
        # f2.writelines(str(targets_all0[:100,:,:]))
        # f1.close()
        # f2.close()

        # print("neuralcube",neuralCube_all0[30, 3000, :],"cube")
        # print("target",targets_all0[:,3000,:],"target")
        # print('finished')

        # imgData = np.zeros((30,3,128,256))

        # list = [x for x in range(0, 192)]
        # plt.figure(figsize=(10, 2))
        #
        # plt.plot(list[0:192],neuralCube_all0[100, 3000, :])
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()


        #一维卷积
        targets_all0 = targets_all[0][:, :, 0:-1]
        targets_all0=targets_all0.transpose(1,0,2)
        neuralCube_all0=neuralCube_all[0].transpose((1,0,2))
        num = 6000
        trainX = neuralCube_all0[:num, :, :]
        trainY = targets_all0[:num, :, :]
        testX = neuralCube_all0[num:, :, ]
        testY = targets_all0[num:, :, :]
        print(trainX.shape,trainY.shape)
        Cov1_model(trainX, trainY, testX, testY)

        # 二维卷积
        # neuralCube_all0 = np.expand_dims(neuralCube_all[0], axis=0)
        # targets_all0 = targets_all[0][:, :, 0:-1]
        # targets_all0=targets_all0.transpose(1,0,2)
        # neuralCube_all0 = np.transpose(neuralCube_all0, (2, 1, 3, 0))
        # num=6000
        # trainX=neuralCube_all0[:num,:,:]
        # trainY=targets_all0[:num,:,:]
        # testX=neuralCube_all0[num:,:,]
        # testY=targets_all0[num:, :, :]
        # lee_mdoel(trainX ,trainY,testX,testY)


    def _loadAllDatasets(self):
        """
        Loads the labels & data for each day specified in the training args, and returns the relevant variables as data cubes.
        Also collects the file names of all .tfrecord files needed for including the synthetic data.
        """
        neuralCube_all = []
        targets_all = []
        errWeights_all = []
        numBinsPerTrial_all = []
        cvIdx_all = []
        recordFileSet_all = []

        for dayIdx in range(self.nDays):
            #print(self.args['labelsFile_' + str(dayIdx)])
            #print(self.args['singleLettersFile_' + str(dayIdx)])
            neuralData, targets, errWeights, binsPerTrial, cvIdx = prepareDataCubesForRNN(
                self.args['sentencesFile_' + str(dayIdx)],
                self.args['singleLettersFile_' + str(dayIdx)],
                self.args['labelsFile_' + str(dayIdx)],
                self.args['cvPartitionFile_' + str(dayIdx)],
                self.args['sessionName_' + str(dayIdx)],
                self.args['rnnBinSize'],
                self.args['timeSteps'],
                self.isTraining)

            neuralCube_all.append(neuralData)
            targets_all.append(targets)
            errWeights_all.append(errWeights)
            numBinsPerTrial_all.append(binsPerTrial)
            cvIdx_all.append(cvIdx)

            synthDir = self.args['syntheticDatasetDir_' + str(dayIdx)]
            if os.path.isdir(synthDir):
                recordFileSet = [os.path.join(synthDir, file) for file in os.listdir(synthDir)]
            else:
                recordFileSet = []

            if self.args['synthBatchSize'] > 0 and len(recordFileSet) == 0:
                sys.exit('Error! No synthetic files found in directory ' + self.args[
                    'syntheticDatasetDir_' + str(dayIdx)] + ', exiting.')

            random.shuffle(recordFileSet)
            recordFileSet_all.append(recordFileSet)

        return neuralCube_all, targets_all, errWeights_all, numBinsPerTrial_all, cvIdx_all, recordFileSet_all



def gaussSmooth(inputs, kernelSD):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.

    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """

    # get gaussian smoothing kernel
    inp = np.zeros([100])
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)

    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # apply the convolution separately for each feature
    convOut = []
    for x in range(192):
        convOut.append(
            tf.nn.conv1d(inputs[:, :, x, tf.newaxis], gaussKernel[:, np.newaxis, np.newaxis].astype(np.float32), 1,
                         'SAME'))

    # gather the separate convolutions together into a 3d tensor again
    smoothedData = tf.concat(convOut, axis=2)

    return smoothedData





# Here we provide support for running from the command line.
# The only command line argument is the name of an args file.
# Launching from the command line is more reliable than launching from within a jupyter notebook, which sometimes hangs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--argsFile', metavar='argsFile',
                        type=str, default='args.p')

    args = parser.parse_args()
    args = vars(args)
    # print(os.getcwd())
    # args='D:/SIAT/DATA/handwritingDATA/RNNTrainingSteps/Step4_RNNTraining/HeldOutTrials/args.p'

    # argDict = pickle.load( open( args['argsFile'], "rb" ) )
    argDict = pickle.load(
        open('D:/SIAT/DATA/handwritingDATA/RNNTrainingSteps/Step4_RNNTraining/HeldOutTrials/args.p', "rb"))

    # set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print('Setting CUDA_VISIBLE_DEVICES to ' + argDict['gpuNumber'])
    os.environ["CUDA_VISIBLE_DEVICES"] = argDict['gpuNumber']

    # instantiate the RNN model
    rnnModel = charSeqRNN(args=argDict)

    # train or infer



