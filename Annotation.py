#In the name of God
import csv
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve
from scipy.io import loadmat

class Annotator:
	def __init__(self):
		self.labels = [[[]]] # A WxHxF video containing labels of the pixels (1 -> foreground, 0 -> background)
	
class JHMDBAnnotator(Annotator):
	def __init__(self, annot_path, video_size=(240,320,30)):
		'''
		:param arg1: path to the annotation file. it should be a csv containing zeros and ones
		:param arg2: size of the video. A tuple -> (height, width, frames)
		:type arg1: string
		:type arg2: tupel
		'''	
		self.labels = [[[0 for i in xrange(30)]for j in xrange(320)] for i in xrange(240)]
		self.__annot_path = annot_path
		self.__video_size = video_size
		mat = loadmat(annot_path)
		self.labels = mat['part_mask']
		'''
		with open(annot_path, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			x = -1
			for row in reader:
				x+=1
				for y in xrange(video_size[1]):
					for f in xrange(video_size[2]):
						self.labels[x][y][f] = int(row[y+f*video_size[1]]) 
		'''
						
import matplotlib.pyplot as plt

class Evaluator:
	@staticmethod
	def AP(dist_ind, gt_labels):
		labels = np.array([1 if gt_labels[r[1]] else 0 for r in dist_ind])
		scores = np.array([r[0] for i in dist_ind])
		return average_precision_score(labels, scores)
	@staticmethod
	def IOI(segment, label):
		pass
	@staticmethod
	def ROCCurve(score, y, linespace=1, clr="r"):
		a,b,c= roc_curve(y, score)
		h = plt.plot(a,b,color=clr)
		return h
		'''
		roc_x = []
		roc_y = []
		min_score = min(score)
		max_score = max(score)
		thr = np.linspace(min_score, max_score, linespace)
		FP=0
		TP=0
		N = sum(y)
		P = len(y) - N
		for (i, T) in enumerate(thr):
		    for i in range(0, len(score)):
			if (score[i] > T):
			    if (y[i]==1):
				TP = TP + 1
			    if (y[i]==0):
				FP = FP + 1
		    roc_x.append(FP/float(N))
		    roc_y.append(TP/float(P))
		    FP=0
		    TP=0
		handle = plt.scatter(roc_x, roc_y, color=clr)
		return handle
		#plt.show()
		'''
