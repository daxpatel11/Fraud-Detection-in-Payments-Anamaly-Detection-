class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True):
        '''
        input:
            lr : Learning Rate
            num_iter : Number of Iterations
            fit_intercept : Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
        '''
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    # add column for intercept
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    # Sigmond Function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        '''
        training function
        input:
            X : dataset
            y : target column/ class column
        '''
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        # Gradient descent
        for _ in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def decision_function(self, X):
        '''
        testing function
        input:
            X : dataset
        '''
        return self.predict_prob(X).round()