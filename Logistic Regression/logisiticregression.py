#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

from classifier import *
import numpy as np



#Note: Here the bias term is considered as the last added feature 

class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001):        
        """
            lembda= Regularization parameter...            
        """
        Classifier.__init__(self,lembda)                
        
        self.lembda = lembda
        self.theta = []
        pass
    def sigmoid(self,z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """

        # Your Code here
        return 1.0/(1.0+2.71828**(-z))
    
    def hypothesis(self, X,theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here
        # hx = []
        # for i in X:
        #     hx.append(self.sigmoid(i.T.dot(theta)))
        # hx = np.array(hx)
        # return hx
        return self.sigmoid(X.dot(theta))

        
    def cost_function(self, X,Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''
    

        # Your Code here
        hx = self.hypothesis(X,theta)
        t1 = Y.T.dot(np.log(hx))
        t2 = (1 - Y.T).dot(np.log(1-hx))
        #t3 = (self.lembda/2.0)*theta.T.dot(theta)
        t4 = ((-t1-t2)/float(X.shape[0]))#+t3
        return float(t4)



    def derivative_cost_function(self,X,Y,theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        
        # Your Code here
        #print (self.hypothesis(X,theta)-Y).shape,"is ki"
        hx = self.hypothesis(X,theta)
        # t1 = (hx-Y).T.dot(X)
        # return (t1/float(X.shape[0])).T# + self.lembda.dot(theta)
        array = []
        t1 = 1.0/X.shape[0]
        for i in xrange(X.shape[1]):
            t2 = np.sum((hx-Y)*X[:,i].reshape(X.shape[0],1))
            array.append(t2/float(X.shape[0]))
        return np.array(array).reshape(X.shape[1],1)


    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''
        
        # Your Code here 
        # Use optimizer here
        self.theta = optimizer.gradient_descent(X,Y,self.cost_function,self.derivative_cost_function)



    
    def predict(self, X):
        
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        
        num_test = X.shape[0]

        
        
        # Your Code here
        res = self.hypothesis(X,self.theta)
        res = np.round(res.astype(np.double))
        return res
        
