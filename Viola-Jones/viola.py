from __future__ import division
import numpy as np
from numpy import cumsum, matlib
import glob
from scipy import misc
import math
import newviolajones



def computeFeature(image, featureByIndex):
    None

class WeakClassifier:
    def __init__(self, alpha, index, theta):
        self.alpha = alpha
        self.index = index

    def hypothesis(self, image):
        return alpha*computeFeature(image,self.index)

class StrongClassifier:
    def __init__(self, T, faces, backgrounds):
        self.faces = faces
        self.backgrounds = backgrounds
        numPos = len(self.faces)
        numNeg = len(self.backgrounds)
        self.numImages = numPos + numNeg
        self.labels = [1 if x < numPos else -1 for x in range(self.numImages)]
        self.weights = [1/(2*numPos) if x < numPos else 1/(2*numNeg) for x in range(self.numImages)]
        weight_sum = sum(self.weights)
        self.weights = [weight/weight_sum for weight in self.weights]
        self.classifiers = []
        self.T = T


    def makeStrongClassifier(self):
        for i in xrange(self.T):
            currentMin, theta, polarity, featureIndex, bestResult = newviolajones.getWeakClassifier(features, self.weights, self.labels, len(faces))
            alpha = math.log(1/(currentMin/(1-currentMin)))
            newClassifier = WeakClassifier(alpha, index)
            self.classifiers.append(newClassifier)
            self.updateWeights(featureIndex, minError, theta)


    def updateWeights(self, featureIndex, minError, threshold):
        B_t = minError/ (1 - minError)
        feature = features[featureIndex,:]
        for i in range(len(self.labels)):
            if polarity > 0:
                if feature[i] < threshold:
                    hyp = polarity
                else:
                    hyp = -polarity
            else:
                if feature[i] > threshold:
                    hyp = polarity
                else:
                    hyp = -polarity
            if hyp != labels[i]:
                self.weights[i]*=B_t

    def classify(image):
        positives = []
        summation = 0
        for weak in self.classifiers:
            summation += weak.hypothesis(image)
        return math.sign(summation)



class ViolaJones:
    def __init__(self):
        self.faces = glob.glob("./test_faces/*.jpg")
        self.backgrounds =  glob.glob("./test_background/*.jpg")
        self.getFeaturesIndices()
        self.cascade = []

    def trainCascade(self, trainingPos, trainingNeg):
        t = 2
        for i in range(5):
            if i > 0:
                self.faces = positives
                self.backgrounds = []
            stage = StrongClassifier(t, self.faces, self.backgrounds)
            cascade.append(stage)
            positives = []
            for image in images:
                if stage.classify(image) > 0:
                    positives.append(image)
            t*= t

    def classifyImage(image):
        for stage in self.cascade:
            if stage.classify(image) < 0:
                return False
        return True


    def getFeaturesIndices(self):
        self.featuresByIndex = []
        window_h = 1; window_w=2 #window/feature size
        for h in xrange(1,row/window_h+1): #extend the size of the rectangular feature
            for w in xrange(1,col/window_w+1):
                for i in xrange (1,row+1-h*window_h+1,4): #stride size=4
                    for j in xrange(1,col+1-w*window_w+1,4):
                        rect1=np.array([i,j,w,h]) #4x1
                        rect2=np.array([i,j+w,w,h])
                        self.featuresByIndex.append([rect1, rect2])


        window_h = 2; window_w=1
        for h in xrange(1,row/window_h+1):
            for w in xrange(1,col/window_w+1):
                for i in xrange (1,row+1-h*window_h+1,4):
                    for j in xrange(1,col+1-w*window_w+1,4):
                        rect2=np.array([i,j,w,h])
                        rect1=np.array([i+h,j,w,h])
                        self.featuresByIndex.append([rect1, rect2])


