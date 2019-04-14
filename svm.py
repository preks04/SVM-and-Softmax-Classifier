import numpy as np
import random

class Svm (object):
    """" Svm classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        st_dev = 0.01
        self.W = st_dev * np.random.randn(inputDim, outputDim)
        pass
        
    def calLoss (self, x, y, reg):
        """
        Svm loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        num_train = x.shape[0]
        score = x.dot(self.W)

        score_yi = score[np.arange(num_train), y]

        margin = score - score_yi[:, np.newaxis] + 1

        loss_i = np.maximum(0, margin)
        loss_i[np.arange(num_train), y] = 0
        loss = np.sum(loss_i) / num_train
        # Loss with regularization
        loss += reg * np.sum(self.W * self.W)
        # Calculating ds
        ds = np.zeros_like(margin)
        ds[margin > 0] = 1
        ds[np.arange(num_train), y] = 0
        count = np.sum(ds, axis=1)
        ds[np.arange(num_train), y] = -count

        dW = (1 / num_train) * (x.T).dot(ds)
        dW = dW + (2 * reg * self.W)

        #
        # score = x.dot(self.W)
        # score_yi = score[np.arange(num_train), y]
        # margin = score - score_yi[:, np.newaxis] + 1
        # loss_i = np.maximum(0, margin)
        # loss_i[np.arange(num_train), y] = 0
        # loss = np.sum(loss_i) / num_train
        #
        # # Loss - L2 regularization
        # loss += reg * np.sum(self.W * self.W)
        #
        # ds = np.zeros_like(margin)
        # ds[margin > 0] = 1
        # ds[np.arange(num_train), y] = 0
        # count = np.sum(ds, axis=1)
        # ds[np.arange(num_train), y] = -count
        #
        # dW = (x.T).dot(ds)
        # dW /= num_train
        # dW = dW + (2 * reg * self.W)

        pass
        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Svm classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            num_train = x.shape[0]
            mask = np.random.choice(num_train, batchSize)
            xBatch = x[mask]
            yBatch = y[mask]

            # evaluating loss and gradient
            loss, dW = self.calLoss(xBatch, yBatch, reg)
            # updating weights
            self.W = self.W - lr * dW
            lossHistory.append(loss)

            # num_train = x.shape[0]
            # mask = np.random.choice(batchSize, num_train)
            # mask1 = np.random.choice(num_train, batchSize)
            # xBatch = x[mask]
            # yBatch = y[mask1]
            #
            #
            # loss, dW = self.calLoss(xBatch, yBatch, reg)
            # lossHistory.append(loss)
            #
            # self.W = self.W - lr * dW

            pass
            
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        score = x.dot(self.W)
        yPred = np.argmax(score, axis = 1)

        pass
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        yPred = self.predict(x)
        acc = np.mean(y == yPred) * 100

        pass
        return acc



