import math
import sys
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def gather_data_plot(trains, tests):
	mistakes_list = []
	for i in range(0, 10):
		mistakes = 0
		plane = [0 for i in range(0, 784)]
		t = 0
		images = [trains[i][0] for i in range(0, len(trains))]
		ans = [trains[i][1] for i in range(0, len(trains))]
		for img in images:
			d = np.dot(plane, img)
			pred = 0
			if(d>0):
				pred = 1
			else:
				pred = -1
			if(pred == -1 and ans[t] == 1):
				plane = np.add(plane, img)
				mistakes = mistakes + 1
			elif(pred == 1 and ans[t] == -1):
				plane = np.subtract(plane, img)
				mistakes = mistakes + 1
			t = t + 1
		mistakes_list.append(mistakes)
	print(mistakes_list)


# Takes in list of (training_image, label) tuples, list of test images
def perceptron(trains, tests):
	# Train
	for i in range(0, 10):
		plane = [0 for i in range(0, 784)]
		t = 0
		images = [trains[i][0] for i in range(0, len(trains))]
		ans = [trains[i][1] for i in range(0, len(trains))]
		for img in images:
			d = np.dot(plane, img)
			pred = 0
			if(d>0):
				pred = 1
			else:
				pred = -1
			if(pred == -1 and ans[t] == 1):
				plane = np.add(plane, img)
			elif(pred == 1 and ans[t] == -1):
				plane = np.subtract(plane, img)
			t = t+1
	# Analyze test cases
	tor_pred = [0 for i in range(0, len(tests))]
	t=0
	for img in tests:
		d = np.dot(plane, img)
		if(d > 0):
			tor_pred[t] = 1
		else:
			tor_pred[t] = -1
		t = t+1
	return tor_pred

def merge_img_label(trains, labels):
	return [(trains[i], labels[i]) for i in range(0, len(trains))]

# Returns set of 1x784 images
def fillImages(file):
	fi = open(file)
	lines = []
	for img in fi:
		pix = img.split()
		lines.append([float(i) for i in pix])
	return lines

# Returns list of +1's or -1's
def fillLabels(file):
	fi = open(file)
	labels = []
	for img in fi:
		labels.append(float(img))
	return labels

def write(text, file):
	f = open(file, 'a')
	f.write(text)
	f.write("\n")
	f.close()

def generateImage(line, finame):
	for i in range(0, len(line)):
		line[i] = line[i] * 255
	img = np.reshape(line, (28,28))
	A = np.array(img)
	out_image = Image.fromarray(A.astype("uint8"))
	out_image.save(finame)


trains = fillImages("train2k.databw.35")
labels = fillLabels("train2k.label.35")
tests = fillImages("test200.databw.35")
trains_tuples = merge_img_label(trains, labels)
tor_pred = perceptron(trains_tuples, tests)
for i in range(0, len(tor_pred)):
	write(str(tor_pred[i]), "perceptron.label.test200")