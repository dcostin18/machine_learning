
import numpy as np
from numpy import cumsum, matlib
import glob
from scipy import misc
import multiprocessing
import math
import time
import json

def getHaar (fyle):
    Nfeatures = 82944 #change this number if you need to use more/less features
    row = 64
    col = 64
    imgGray = misc.imread (fyle, flatten=1) #array of floats, gray scale image
            # convert to integral image
    intImg = np.zeros((row+1,col+1))
    intImg [1:row+1,1:col+1] = np.cumsum(cumsum(imgGray,axis=0),axis=1)
            # compute features
    return computeFeature(intImg,row,col,Nfeatures)

def sumRect(I, rect_four):

    row_start = rect_four[0]
    col_start = rect_four[1]
    width = rect_four[2]
    height = rect_four[3]
    one = I[row_start-1, col_start-1]
    two = I[row_start-1, col_start+width-1]
    three = I[row_start+height-1, col_start-1]
    four = I[row_start+height-1, col_start+width-1]
    rectsum = four + one -(two + three)
    return rectsum

def computeFeature(I, row, col, numFeatures):
    feature = np.zeros(numFeatures)

    #extract horizontal features
    cnt = 0 # count the number of features
    # This function calculates cnt=295937 features.
    window_h = 1; window_w=2 #window/feature size
    for h in xrange(1,row/window_h+1): #extend the size of the rectangular feature
        for w in xrange(1,col/window_w+1):
            for i in xrange (1,row+1-h*window_h+1,8): #stride size=4
                for j in xrange(1,col+1-w*window_w+1,8):
                    rect1=np.array([i,j,w,h]) #4x1
                    rect2=np.array([i,j+w,w,h])
                    feature [cnt]=sumRect(I, rect2)- sumRect(I, rect1)
                    cnt=cnt+1

    window_h = 2; window_w=1
    for h in xrange(1,row/window_h+1):
        for w in xrange(1,col/window_w+1):
            for i in xrange (1,row+1-h*window_h+1,8):
                for j in xrange(1,col+1-w*window_w+1,8):
                    rect1=np.array([i,j,w,h])
                    rect2=np.array([i+h,j,w,h])
                    feature[cnt]=sumRect(I, rect1)- sumRect(I, rect2)
                    cnt=cnt+1
    return feature

if __name__=="__main__":
    Nfeatures = 82944
    start_time = time.time()
    features = np.zeros((Nfeatures, 2000))
    pool = multiprocessing.Pool(4)
    faces = glob.glob("./faces/*.jpg")[:1000]
    backgrounds = glob.glob("./background/*.jpg")[:1000]
    files = faces + backgrounds
    print len(files)
    print len(features[0])
    feature_array =  pool.map(getHaar,files)
    print len(feature_array)
    for i in range(len(feature_array)):
        features[:,i] = feature_array[i]
    print "computed"
    print("--- %s seconds ---" % (time.time() - start_time))
    disk = open('80k_features.json', 'w')
    json.dump(features.tolist(), disk)
    # disk = open('test_features.json', 'r+')
    # loaded = np.array(json.load(disk))
    # print "feat ", features[1000,:]
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print "load ", loaded[1000,:]




