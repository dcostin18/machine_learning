import math
import sys
import random
from PIL import Image
import numpy as np

def fillLabels(file):
	fi = open(file)
	labels = []
	for img in fi:
		labels.append(float(img))
	return labels

def gaussian(trains_tuples, tests):
	# TRAINING
	# Make list of all -1's and 1's
	pts_pos = []
	pts_neg = []
	mean_pos = [0 for i in range(0, 784)]
	mean_neg = [0 for i in range(0, 784)]
	for i in range(0, len(trains_tuples)):
		if(trains_tuples[i][1] == 1):
			pts_pos.append(trains_tuples[i][0])
			mean_pos = np.add(mean_pos, trains_tuples[i][1])

		else:
			pts_neg.append(trains_tuples[i][1])
			mean_neg = np.add(mean_neg, trains_tuples[i][1])
	mean_pos = [mean_pos[i]/len(pts_pos) for i in range(0, 784)]
	mean_neg = [mean_neg[i]/len(pts_neg) for i in range(0, 784)]
	# Sigma calculations
	sigma_pos = np.empty((784,784))
	for i in range(0, len(pts_pos)):
		v = np.subtract(pts_pos[i], mean_pos)
		sigma_pos = sigma_pos + np.dot(v,v)
		sigma_pos = np.add(sigma_pos, np.outer(v,v))
	sigma_pos = [sigma_pos[i]/len(pts_pos) for i in range(0, len(sigma_pos))]	
	sigma_neg = np.empty((784,784))
	for i in range(0, len(pts_neg)):
		v = np.subtract(pts_neg[i], mean_neg)
		sigma_neg = np.add(sigma_neg, np.outer(v,v))
	sigma_neg = [sigma_neg[i]/len(pts_neg) for i in range(0, len(sigma_neg))]
	print(mean_pos[:7], mean_neg[:7], sigma_pos[:7], sigma_neg[:7])
	print(len(pts_pos), len(pts_neg))

	# TESTING
	tor = []
	for v in tests:
		reg = np.identity(784) * 0.01
		sigma_pos = np.add(sigma_pos, reg)
		sigma_neg = np.add(sigma_neg, reg)
		a,b = np.linalg.eig(sigma_pos)
		c,d = np.linalg.eig(sigma_neg)
		log_sigma_pos = sum([math.log(a[i]) for i in range(len(a))])
		log_sigma_neg = sum([math.log(c[i]) for i in range(len(c))])
		pi = len(pts_pos) / len(pts_neg)
		one = math.log(pi) - math.log(abs(log_sigma_pos))/2 - np.dot(np.dot(np.transpose(np.subtract(v, mean_pos)),np.linalg.inv(sigma_pos)), np.subtract(v,mean_pos)/2)
		two = math.log(1 - pi) - math.log(abs(log_sigma_pos))/2 - np.dot(np.dot(np.transpose(np.subtract(v, mean_neg)), np.linalg.inv(sigma_neg)), np.subtract(v,mean_neg)/2)
		r = one - two
		if(r>=0):
			tor.append(1)
		else:
			tor.append(-1)
	return tor

def fillImages(file):
	fi = open(file)
	lines = []
	for img in fi:
		pix = img.split()
		lines.append([float(i) for i in pix])
	return lines

def merge_img_label(trains, labels):
	return [(trains[i], labels[i]) for i in range(0, len(trains))]

def generateImage(line, finame):
	img = [[[] for i in range(0, 28)] for x in range(0, 28)]
	for row in range(0, 28):
		for col in range(0, 28):
			img[row][col] = line[28*row + col] * 255
	A = np.array(img)
	out_image = Image.fromarray(A.astype("uint8"))
	out_image.save(finame)

trains = fillImages("train2k.databw.35")
tests = fillImages("test200.databw.35")
labels = fillLabels("train2k.label.35")
trains_tuples = merge_img_label(trains, labels)
print(gaussian(trains_tuples, tests))
