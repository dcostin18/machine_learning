import math
import sys
import random
from PIL import Image
import numpy as np
from numpy import linalg as LA

# Pass unscaled images
def lfa(lines, k):
	m = mean(lines)
	# Center data
	for img in range(0, 5000):
		for pix in range(0, 784):
			lines[img][pix] = lines[img][pix] - m[pix]
	# Scale data
	for img in range(0, 5000):
		lines[img] = rescale(lines[img])
	cov = np.cov(lines, rowvar = 0)
	eigen_vals, eigen_vecs = LA.eig(cov)
	print(eigen_vals[:10])
	for i in range(0, 10):
		eig_vec = eigen_vecs[:,i]
		generateImage(eig_vec, "basis_"+str(i)+".tiff")

# Takes in unscaled images
def mean(lines):
	m = [0] * 784
	for img in range(0, len(lines)):
		for pix in range(0, 784):
			m[pix] = m[pix] + lines[img][pix]
	for i in range(0, 784):
		m[i] /= 5000
	return rescale(m)

def rescale(img):
	minim = 0
	maxim = 0
	for pix in range(0, 784):
		n = img[pix]
		if n<minim:
			minim = n
		elif n > maxim:
			maxim = n
	for pix in range(0, 784):
		img[pix] = ((img[pix] - minim)/(maxim - minim)) * 255
	return img 

def generateImage(line, finame):
	img = [[[] for i in range(0, 28)] for i in range(0, 28)]
	for row in range(0, 28):
		for col in range(0, 28):
			img[row][col] = line[28*row + col]
	A = np.array(img)
	out_image = Image.fromarray(A.astype("uint8"))
	out_image.save(finame)


def fill(file):
	fi = open(file)
	# Read in as 5000x784 matrix:
	lines = []
	for img in fi:
		pix = img.split()
		lines.append([float(i) for i in pix])
	# Rescale to [0,1]
	# for img in range(0, 5000):
		# lines[img] = rescale(lines[img])
	return lines

lines = fill("lfa.txt")
generateImage(rescale(lines[176]), "LINES0.tiff")
generateImage(mean(lines), "mean.tiff")
lfa(lines, 5000)