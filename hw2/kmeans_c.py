import math
import sys
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy

def kmeans_algo(pts, const_k):
	it = 0
	n = len(pts)
	# Chose initial centers at random
	# Note that cluster centers are colors
	centers = []
	for i in range(0, const_k):
		centers.append(pts[random.randint(0,127)][random.randint(0,127)])
	cond = False
	while cond == False:
		# print(it)
		# it = it+1
		# Instantiate empty clusters array with const_k slots/groups
		clusters = [[] for i in range(0, const_k)]
		# For each point, assign to closest centered cluster:
		for row in range(0, len(pts)):
			for col in range(0, len(pts[0])):
				# For each pixel [row, col]:
				mindist = -1
				minj = -1
				# For each cluster
				for j in range(0, const_k):
					dist = color_distance(pts[row][col], centers[j])
					if(mindist == -1 or dist < mindist):
						mindist = dist
						minj = j
				clusters[minj].append([row, col])
		centers2 = []
		for i in range(0, const_k):
			centers2.append(color_normalize(clusters[i], pts))
		# print('centers: ', centers[:2], 'centers2', centers2[:2])
		if(centers == centers2):
			cond = True
		centers = centers2
	return (clusters, centers)

def color_normalize(clust, pts):
	# Returns new center: average of all colors in cluster
	center = [0,0,0]
	for i in range(0, len(clust)):
		center = [x+y for x,y in zip(center, pts[clust[i][0]][clust[i][1]])]
	return [x / len(clust) for x in center]

def color_distance(col1, col2):
	return math.sqrt(math.pow(col1[0] - col2[0], 2) + math.pow(col1[1] - col2[1],2) + math.pow(col1[2] - col2[2], 2))

# Finding "average location" of list of 2d points
def normalize(ls):
	xsum_norm = sum([a[0] for a in ls]) / float(len(ls))
	ysum_norm = sum([a[1] for a in ls]) / float(len(ls))
	return [xsum_norm, ysum_norm]

def fill_image():
	in_image = Image.open("bird_small.tiff")
	A = numpy.asarray(in_image)
	return A.tolist()

def generateImage(clusts, centers, rows):
	# Fill image by color cluster
	img = [[[] for r in range(0, rows)] for i in range(0, rows)]
	# For each cluster
	for c in range(0, len(clusts)):
		# For each point in cluster
		for p in range(0, len(clusts[c])):
			pt = clusts[c][p]
			img[pt[0]][pt[1]] = centers[c]
	# Generate acutal image
	A = numpy.array(img)
	out_image = Image.fromarray(A.astype("uint8"), "RGB")
	out_image.save("output-bird.tiff")

def fill(file):
	fi = open(file)
	ls = []
	for line in fi:
		point = line.split("\t")
		ls.append([float(point[0]), float(point[1][:-1])])
	fi.close()
	return ls

def distance(p1, p2):
	return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

pts = fill_image()
(clusts, centers) = kmeans_algo(pts, 16)
generateImage(clusts, centers, 128)

# points_ls = fill(finame)
# clust = kmeans_algo(points_ls, 4)

# plt.plot([pt[0] for pt in clust[0]], [pt[1] for pt in clust[0]], 'ro')
# plt.plot([pt[0] for pt in clust[1]], [pt[1] for pt in clust[1]], 'go')
# plt.plot([pt[0] for pt in clust[2]], [pt[1] for pt in clust[2]], 'bo')
# plt.plot([pt[0] for pt in clust[3]], [pt[1] for pt in clust[3]], 'yo')
# plt.show()
