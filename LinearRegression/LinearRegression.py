import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        print("x.shape {}, y.shape{}".format(x.shape,y.shape))
        w = self.W
        for i in range(epochs):
            loss = 0
            s = np.arange(x.shape[0])
            np.random.shuffle(s)
            x = x[s]
            y = y[s]
            wd = np.zeros_like(self.W)
            for j in range(0, x.shape[0], batch_size):
                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
                y_predicted  = self.forward(x_batch)
                loss += np.sum((y_predicted - y_batch) ** 2)
                wd = sum((y_predicted - y_batch)*2*x_batch/len(x_batch)).reshape(self.num_features, 1)
                w = optim.update(w, wd, lr)
                self.W = w
            loss = loss / (x.shape[0])
        print ("cost {}, batch_size {}, epoch {}".format(loss,batch_size,epochs))
        final_loss = loss
        # ============================================================
        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # ========================= EDIT HERE ========================
        w = self.W
        y_predicted = 0
        y_predicted = np.dot(x, w)
        #print(x)
        #print(y_predicted)
        
        # ============================================================
        return y_predicted
