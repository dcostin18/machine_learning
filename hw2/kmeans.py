import math
import sys
import random
import matplotlib.pyplot as plt

finame = sys.argv[1]

def kmeans_algo(pts, const_k):
	n = len(pts)
	# Chose initial centers at random
	centers = random.sample(pts, const_k)
	cond = False
	while cond == False:
		# Instantiate empty clusters array
		clusters = [[] for i in range(0, const_k)]
		# For each point, assign to closest centered cluster:
		for i in range(0, n):
			mindist = -1
			minj = -1
			# For each cluster:
			for j in range(0, const_k):
				dist = distance(pts[i], centers[j])
				if(mindist == -1 or dist < mindist):
					mindist = dist
					minj = j
			clusters[minj].append(pts[i])
		# Add the centers to their respective clusters
		for i in range(0, const_k):
			clusters[i].append(centers[i])
		# Recompute cluster centers
		centers2 = []
		for i in range(0, const_k):
			centers2.append(normalize(clusters[i]))
		if(centers == centers2):
			cond = True
		centers = centers2
	return clusters

# Finding "average location" of list of 2d points
def normalize(ls):
	xsum_norm = sum([a[0] for a in ls]) / float(len(ls))
	ysum_norm = sum([a[1] for a in ls]) / float(len(ls))
	return [xsum_norm, ysum_norm]


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

points_ls = fill(finame)
clust = kmeans_algo(points_ls, 4)

plt.plot([pt[0] for pt in clust[0]], [pt[1] for pt in clust[0]], 'ro')
plt.plot([pt[0] for pt in clust[1]], [pt[1] for pt in clust[1]], 'go')
plt.plot([pt[0] for pt in clust[2]], [pt[1] for pt in clust[2]], 'bo')
plt.plot([pt[0] for pt in clust[3]], [pt[1] for pt in clust[3]], 'yo')
plt.show()
