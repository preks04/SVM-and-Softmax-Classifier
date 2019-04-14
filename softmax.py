import numpy as np


class Softmax (object):
    """" Softmax classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        st_dev = 0.01
        self.W = st_dev * np.random.randn(inputDim, outputDim)

        pass

    def calLoss (self, x, y, reg):
        """
        Softmax loss function
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
        score_n = score - np.max(score, axis=1, keepdims = True)
        exp_score = np.exp(score_n)
        sum_f = np.sum(exp_score, axis=1, keepdims = True)

        # calculating probability of incorrect label
        p = exp_score / sum_f
        p_yi = exp_score[np.arange(num_train), y] / sum_f

        # calculating loss by applying log over the probability
        loss_i = -np.log(p_yi)

        loss = np.sum(loss_i) / num_train
        loss += reg * np.sum(self.W * self.W)
        ds = p.copy()
        ds[np.arange(num_train), y] += -1
        dW = x.T.dot(ds)
        dW /= num_train
        dW += 2 * reg * self.W

        pass
        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Softmax classifier using stochastic gradient descent.
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
        A list containing the value of the loss function at each training iteration.
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
            lossHistory.append(loss)
            # updating weights
            self.W = self.W - lr * dW

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
        yPred = np.argmax(score, axis=1)

        pass
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        yPred = self.predict(x)
        acc = np.mean(y == yPred) * 100

        pass
        return acc



