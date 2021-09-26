import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch
        #self.W = np.array([[-0.6317466601918094], [-0.05653350853635963], [0.8556811494570249], [-0.03946554989032114], [-0.8449024756038757], [0.2661752869088194], [0.002126179261684902]]) 
        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        print("x.shape {}, y.shape{}".format(x.shape,y.shape))
        w = self.W
        for i in range(epochs):
            loss = 0.0
            s = np.arange(x.shape[0])
            np.random.shuffle(s)
            x = x[s]
            y = y[s]
            wd = np.zeros_like(self.W)
            for j in range(0, x.shape[0], batch_size):
                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
                y_predicted = self._sigmoid(x_batch)
                loss -= np.sum(y_batch * np.log(y_predicted + epsilon) + (1 - y_batch) * np.log(1 - y_predicted + epsilon))
                wd = sum((y_predicted - y_batch) * x_batch/len(x_batch)).reshape(self.num_features, 1)
                w = optim.update(w, wd, lr)
                self.W = w
            loss = loss/x.shape[0]
        print ("cost {}, batch_size {}, epoch {}".format(loss,batch_size,epochs))
        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        y_predicted = self._sigmoid(x)
        for i in range(0, y_predicted.shape[0]):
            if y_predicted[i] >= threshold:
                y_predicted[i] = 1
            else:
                y_predicted[i] = 0

        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        w = self.W
        y_predicted = np.dot(x,w)
        sigmoid = 1.0/(1.0 + np.exp(-y_predicted))
        # ============================================================
        return sigmoid
