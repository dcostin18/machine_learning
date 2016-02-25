from __future__ import division
import numpy as np
from numpy import cumsum, matlib
import glob
from scipy import misc
import math
import PIL
import time
import newviolajones
import archive5
import json

disk = open('featuresByIndex.json', 'r+')
featuresByIndex = np.array(json.load(disk))
print "NumFeatures", len(featuresByIndex)

def evalFeature(index, image):
    row = 64
    col = 64
    # imgGray = misc.imread(image, flatten=1)
    intImg = np.zeros((row+1,col+1))
    intImg [1:row+1,1:col+1] = np.cumsum(cumsum(image,axis=0),axis=1)
    rects = featuresByIndex[index]
    return archive5.sumRect(intImg, rects[0]) - archive5.sumRect(intImg, rects[1])

class WeakClassifier():
    def __init__ (self, index, theta, polarity, alpha):
        self.index = index
        self.theta = theta
        self.polarity = polarity
        self.alpha = alpha

    def hypothesis(self, image):
        hx = self.polarity*(evalFeature(self.index,image) - self.theta)
        return 1 if hx >= 0 else -1

    def linearVal(self,image):
        return (self.alpha * self.hypothesis(image))


class StrongClassifier:
    def __init__(self, T, faces, backgrounds, features, weights, labels):
        self.faces = faces
        self.backgrounds = backgrounds
        self.images = self.faces + self.backgrounds
        self.numImages = len(faces) + len(backgrounds)
        self.labels = labels
        self.weights = weights
        self.features = features
        self.classifiers = []
        self.T = T
        self.Omega = 0
        self.results = [0]*self.numImages

    def makeStrongClassifier(self):
        start_time = time.time()
        for t in range(self.T):
            currentMin, theta, polarity, featureIndex, bestResult = archive5.getWeakClassifier(self.features, self.weights, self.labels, len(self.faces))
            t+=1
            if currentMin <= 0:
                beta_t, alpha = 1,1
            else:
                beta_t = currentMin/(1 - currentMin)
                if beta_t <= 0:
                    alpha = 1
                    beta_t = 0
                else:
                    alpha = math.log(1/beta_t)
            print "currentMin is", currentMin
            print "Alpha is", alpha
            newClassifier = WeakClassifier(featureIndex, theta , polarity, alpha)
            self.classifiers.append(newClassifier)
            self.updateWeights(beta_t, bestResult)
            for i in range(self.numImages):
                if bestResult[i] == 0:
                    self.results[i] -= alpha
                else:
                    self.results[i] += alpha
            self.Omega = min(self.results[:len(self.faces)])

            false_pos = 0
            false_neg = 0
            indices = []
            for index, res in enumerate(self.results):
                if res >= 0:
                    indices.append(index)
                    if self.labels[index] == 0:
                        false_pos +=1
                elif self.labels[index] == 1:
                    false_neg +=1
            print "Omega is", self.Omega
            false_negative = false_neg/(false_neg + len(self.faces))
            if len(self.backgrounds) == 0:
                false_positive = 0
            else:
                false_positive = false_pos/(false_pos + len(self.backgrounds))
            print "False neg rate is", false_negative
            print "False positive rate is ", false_positive
            if false_negative == 0 and false_positive <= 0.2:
                break
        print("--- %s seconds ---" % (time.time() - start_time))
        return np.array(indices)



    def updateWeights(self, beta_t, bestResult):
        for i in range(len(self.labels)):
            if bestResult[i] == self.labels[i]:
                self.weights[i]*=beta_t

    def classify(self,image,label):
        summation = 0
        for weakClassifier in self.classifiers:
            summation += weakClassifier.linearVal(image)
        if label == 1:
            if summation >= self.Omega:
                return 1
            else:
                return -1
        else:
            if summation >= 0:
                return 1
            else:
                return -1


class ViolaJones:
    def __init__(self):
        self.faces = glob.glob("./faces/*.jpg")
        self.backgrounds =  glob.glob("./background/*.jpg")
        numPos = len(self.faces)
        numNeg = len(self.backgrounds)
        self.numImages = numNeg + numPos
        self.cascade = []
        self.labels = np.array([1 if x < numPos else 0 for x in range(self.numImages)])
        self.weights = [1/(2*numPos) if x < numPos else 1/(2*numNeg) for x in range(self.numImages)]
        weight_sum = sum(self.weights)
        self.weights = np.array([weight/weight_sum for weight in self.weights])

#Need to feed new weights, labels, and features into the new stage.
    def trainCascade(self):
        ts = [2,4,8,10,12,16]
        for i in range(6):
            if i > 0:
                self.faces = positives
                self.backgrounds = negatives
                self.labels = newLabels
                numPos = len(self.faces)
                numNeg = len(self.backgrounds)
                self.numImages = numNeg + numPos
                self.weights = [1/(2*numPos) if self.labels[x] > 0 else 1/(2*numNeg) for x in range(self.numImages)]
                weight_sum = sum(self.weights)
                self.weights = np.array([weight/weight_sum for weight in self.weights])
                self.features = new_features
            else:
                disk = open('80k_features.json', 'r+')
                print "Read features"
                self.features = np.array(json.load(disk))
            stage = StrongClassifier(ts[i], self.faces, self.backgrounds, self.features, self.weights, self.labels)
            indices = stage.makeStrongClassifier()
            self.cascade.append(stage)
            positives = []
            negatives = []
            images = self.faces + self.backgrounds
            for index,val in enumerate(indices):
                if self.labels[val] > 0:
                    positives.append(images[val])
                else:
                    negatives.append(images[val])
            print "Total are", len(positives) + len(negatives)
            indices = np.array(indices)
            new_features = np.zeros((82944, len(indices)))
            newLabels = np.array([0]*len(indices))
            for index,i in enumerate(indices):
                newLabels[index] = self.labels[i]
            sortedLabels = np.argsort(newLabels)
            newLabels = np.sort(newLabels)[::-1]
            sortedIndices = indices[sortedLabels][::-1]
            for index, i in enumerate(sortedIndices):
                new_features[:,index] = self.features[:,i]


    def classifyImage(self,image):
        for stage in self.cascade:
            if stage.classify(image,0) <= 0:
                return False
        return True

def drawSquare(i,j,imgGray):
    width = 1600 - j
    height = 1280 - i
    for col in range(min(64,height)):
        try:
            imgGray[i][j+col] = 255
            imgGray[min(i+63, 1279)][j+col] = 255
        except IndexError:
            "Err"
            None
            #print width, height
    for row in range(min(64,width)):
        try:
            imgGray[i+row][j] = 255
            imgGray[i+row][min(j+63, 1599)] = 255
        except IndexError:
            "Err"
            None
            #print width,height

def classifyTestImage(cascade):
    image = glob.glob("./TestImage.jpg")
    imgGray = misc.imread(image[0], flatten=1)
    k, p = imgGray.shape
    print k,p
    strideLen = 10
    width = 1280
    length = 1600
    iters = 0
    for i in range(0,width,10):
        for j in range(0,length,10):
            window = np.zeros((64,64))
            for row in range(min(64, width - i)):
                for col in range(min(64, length - j)):
                    try:
                        window[row][col] = imgGray[i+row][j+col]
                    except IndexError:
                        print "Errrrrr"
            if cascade.classifyImage(window):
                drawSquare(i,j,imgGray)
    out = PIL.Image.fromarray(imgGray.astype("uint8"))
    out.save('new_squared.jpg')




if __name__=="__main__":
    # image = glob.glob("./TestImage.jpg")
    # imgGray = misc.imread(image[0], flatten=1)
    # for i in range(0,1600,10):
    #     drawSquare(0,i,imgGray)
    # out = PIL.Image.fromarray(imgGray.astype("uint8"))
    # out.save('squared.jpg')
    algo = ViolaJones()
    algo.trainCascade()
    classifyTestImage(algo)
    # faces = glob.glob("./test_faces/*.jpg")
    # print algo.classifyImage(faces[0])
