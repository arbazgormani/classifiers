#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
from numpy import inf
import random

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:


        """
        #print "   "
        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#

    #------------ Calculate Split Score of [1,2,3] classes --------#
    def calculateSplitScore(self,N,n):
        total1 = float(sum(N))
        dy = 0
        for i in N:
            temp = i/total1
            if temp == 0:
                temp = 1
            dy = dy + -(i/float(total1))*np.log2(temp)
        N2 = n - N
        total2 = float(sum(N2))
        dn = 0
        for j in N2:
            temp = j/total2
            if temp == 0:
                temp = 1
            dn = dn + -(j/float(total2))*np.log2(temp)
        return (total1/float(total1+total2))*dy + (total2/float(total1+total2))*dn
        
    def evaluate_numerical_attribute(self, feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''

        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        #         n = []
        n = np.zeros((classes.shape[0]))
        N = []
#         midPoints = (f[1:] + f[:-1]) / 2
        midPoints = []
        for j in range(0,sY.shape[0]-1):
            n[classes == sY[j]] += 1
#             print n
            if (f[j+1] != f[j]):
                m = (f[j+1] + f[j]) / 2    # Mid Point between to adjacent points
                midPoints.append(m)  
                N.append(np.array(n))
        n[classes == sY[sY.shape[0]-1]] += 1
        N = np.array(N)
        e = []
#         print N
        for i in N:
            e.append(self.calculateSplitScore(i,n))
        e = np.array(e)
        ind = np.argmin(e)    # minimum split score index
        split = midPoints[ind]    # Best split point  
        mingain = e[ind]        # Best Split Score
        xlidx = np.argwhere(f<split)
        xridx = np.argwhere(f>split)
        Xlidx = xlidx.reshape((xlidx.shape[0],))
        Xridx = xridx.reshape((xridx.shape[0],))
        #---------End of Your Code-------------------------#

        return split,mingain,Xlidx,Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using
        a random set of features from the given set of features...

    """

    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        self.fidx = -1
        self.splitPoint = 0
        
        #pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        #print "Inside the train of Random"
        nexamples,nfeatures=X.shape

        #print "Train has X of length ", X.shape


        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        #for i in range(self.nrandfeat):
        randfeat = random.sample(range(0,nfeatures),self.nrandfeat)
        # randfeat = randfeat.astype(int)
        for i in randfeat:    
            mingain = 9999
            splitvalue, minscore, xlidx, xridx = self.findBestRandomSplit(X[:,i],Y)
            if (mingain > minscore):
                self.fidx = randfeat
                mingain = minscore
                self.splitPoint = split = splitvalue
                Xlidx = xlidx
                Xridx = xridx

        return split, mingain, Xlidx, Xridx

        #---------End of Your Code-------------------------#
        #return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """

            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)
        max = np.max(feat)
        min = np.min(feat)


        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        minscore = 9999
        splitvalue = 0
        for i in range(self.nsplits):
            randSplit = np.random.uniform(min,max)
            boolIndices = feat<randSplit
            splitScore = self.calculateEntropy(Y,boolIndices)
            if (minscore > splitScore):
                minscore = splitScore
                splitvalue = randSplit
                Xlidx = boolIndices
                Xridx = np.logical_not(boolIndices)
        
            
            
        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx

    def calculateEntropy(self, Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl))
        hr= -np.sum(pr*np.log2(pr))

        sentropy = pleft * hl + pright * hr

        return sentropy


    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        print X,self.fidx
        return (X[self.fidx] < self.splitPoint)
        #---------End of Your Code-------------------------#




# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self,nsplits)
        self.a = None
        self.b = None
        self.c = None

        #pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible

            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        self.fidx = random.sample(range(0, nfeatures), 2)
        alist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        blist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        clist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        minscore = 999
        for i in range(self.nsplits):
            result = alist[i]*X[:,self.fidx[0]] + blist[i]*X[:,self.fidx[1]] + clist[i]
            boolIndices = result>0
            splitScore = self.calculateEntropy(Y,boolIndices)
            if (minscore > splitScore):
                self.a = alist[i]
                self.b = blist[i]
                self.c = clist[i]
                minscore = splitScore
                bXl = boolIndices
                bXr = np.logical_not(boolIndices)
                split = minscore
                

            

        #---------End of Your Code-------------------------#
        return split, minscore, bXl, bXr
        # return minscore, bXl, bXr



    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if ((self.a*X[self.fidx[0]] + self.b*X[self.fidx[1]] + self.c) > 0):
            return True
        else:
            return False
        #---------End of Your Code-------------------------#


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.e = None
        self.f = None

        #pass

    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        self.fidx = random.sample(range(0, nfeatures), 2)
        alist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        blist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        clist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        dlist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        elist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)
        flist = np.random.uniform(-3,3,self.nsplits)#random.sample(range(-3, 3), self.nsplits)

        minscore = 999
        for i in range(self.nsplits):
            x = X[:,self.fidx[0]]
            y = X[:,self.fidx[1]]

            result = alist[i]*x**2 + blist[i]*y**2 + clist[i]*x*y + dlist[i]*x + elist[i]*y + flist[i]
            boolIndices = result>0
            splitScore = self.calculateEntropy(Y,boolIndices)
            if (minscore > splitScore):
                self.a = alist[i]
                self.b = blist[i]
                self.c = clist[i]
                self.d = dlist[i]
                self.e = elist[i]
                self.f = flist[i]
                minscore = splitScore
                bXl = boolIndices
                bXr = np.logical_not(boolIndices)
                split = minscore

        #---------End of Your Code-------------------------#
        return split, minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        x = X[self.fidx[0]]
        y = X[self.fidx[1]]
        if ((self.a*x**2 + self.b*y**2 + self.c*x*y + self.d*x + self.e*y + self.f) > 0):
            return True
        else:
            return False

        #---------End of Your Code-------------------------#