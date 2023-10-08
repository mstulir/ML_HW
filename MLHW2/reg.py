import numpy as np

class LogisticRegression:
    def __init__(self, d):
        self.w = np.random.randn(d) # returns an array of d items

    def compute_loss(self, X, Y):
        """
        Compute l(w) with n samples.
        Inputs:
            X  - A numpy matrix of size (n, d). 1000x10 Each row is a sample.
            Y  - A numpy array of size (n,). n=1000 Each element is 0 or 1.
        Returns:
            A float.
        """
        d=X.shape[1]
        n=X.shape[0]
        loss=0
        for i in range(n):
            loss+=np.log(1+np.exp(np.dot(self.w,X[i,:])))-Y[i]*np.dot(self.w,X[i,:])

        loss=loss/n
        return loss

    def compute_grad(self, X, Y):
        """
        Compute the derivative of l(w).
        Inputs: Same as above.
            X  - A numpy matrix of size (n, d). Each row is a sample.
            Y  - A numpy array of size (n,). Each element is 0 or 1.
        Returns:
            A numpy array of size (d,). 10
        """
        d=X.shape[1]
        n=int(X.shape[0])
        #make array of length d to put values into
        grad=(1/n)*np.dot(X.T,(np.exp(np.dot(X,self.w))/(1+np.exp(np.dot(X,self.w)))-Y))
        return grad

    def train(self, X, Y, eta, rho):
        """
        Train the model with gradient descent.
        Update self.w with the algorithm listed in the problem.
        Returns: Nothing.
        """
        run=True
        i=0
        while run:
            i+=1

            grad=self.compute_grad(X,Y)

            euclidean_distance = np.linalg.norm(grad)

            if euclidean_distance<rho:
                run=False
            self.w=self.w-(eta*grad)




if __name__ == '__main__':
    # Sample Input/Output
    d = 10
    n = 1000

    np.random.seed(0)
    X = np.random.randn(n, d) # n=1000 x d=10 -- matrix
    Y = np.array([0] * (n // 2) + [1] * (n // 2)) #Y is an array of values length n=1000
    eta = 1e-3
    rho = 1e-6

    reg = LogisticRegression(d) # d is an array of 10 items

    reg.train(X, Y, eta, rho)


    # The output should be close to
    # [ 0.15289573 -0.063752   -0.06434498 -0.02005378  0.07812127 -0.04307333
    #  -0.0691539  -0.02769485 -0.04193284 -0.01156307] (10 items)
    # Error should be less than 0.001 for each element
