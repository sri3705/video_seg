#In the name of God
from Supervoxel import *
import caffe
import cPickle as pickle
from scipy.spatial.distance import euclidean
#from Annotation import JHMDBAnnotator as annot
import heapq
from Segmentation import * #MySegmentation

class EmbeddingTester:
	def __init__(self, segment):
		self.segment = segment
		self.segment.__class__ = MySegmentation

	def loadModel(self, snapshot_path, net_path, test_model_path=None):
		self.net_path = net_path
		self.snapshot_path = snapshot_path
		print 'model'
		self.model = caffe.Net(net_path, snapshot_path, caffe.TEST)
		print 'test_model'
		if test_model_path:
			self.test_model = caffe.Net(test_model_path, snapshot_path, caffe.TEST)

	
#TODO: implement features
#	dataset_path.format(name='segmentors_lvl1.p')
	def loadSegment(self, segment_pickle_path, features_pickle_path=None):
		self.segments = pickle.load(open(segment_pickle_path, 'r'))
		self.segment = self.segment[0]
		#features = pickle.load(open(dataset_path.format(name='features_lvl1.p'), 'r'))
		

	def getFeaturesOfSegment(self, seg_num):
		self.model.test_nets[0].forward()	

	#TODO implement this
	def computeDistanceMatrix(self):
		'''
		computes distance between every pair of supervoxels in segment and stores it
		in dist[][] and return
		'''
		#seg = self.segments[0]
		features = self.__extractFeatures(self.segment)
		self.__prepare(self.test_model, self.model, features)
		n = len(self.segment.supervoxels_list)
		dist_mat = np.zeros((n,n))#[[0 for i in xrange(n)] for j in xrange(n)]
		representations = self.getRepresentations(self.test_model)
		dist_mat[...] = -1 * (representations.dot(representations.T))
#		for target_id in range(len(self.segment.supervoxels_list)):
#			for i in range(representations.shape[0]):
#				dist_mat[target_id][i] = -1*np.dot(representations[i][...], representations[target_id][...])
		return dist_mat

	def computeBaselineDistanceMatrix(self):
		'''
		computes distance between every pair of supervoxels in segment and stores it
		in dist[][] and return
		'''

		from scipy.spatial.distance import euclidean
		#seg = self.segments[0]
		features = self.__extractFeatures(self.segment)
		#self.__prepare(self.test_model, self.model, features)
		n = len(self.segment.supervoxels_list)
		dist_mat = np.zeros((n,n))#[[0 for i in xrange(n)] for j in xrange(n)]
		representations = np.array(features['target'])#self.getRepresentations(self.test_model)
		#TODO: Euclidean or Dot?
		dist_mat[...] = -1 * (representations.dot(representations.T))

		#for j in xrange(n-1):
		#	for i in xrange(j+1, n):
		#		dist_mat[i][j] = dist_mat[j][i] = euclidean(representations[j][...], representations[i][...])
				
		return dist_mat


	def retrieveKNearest(self, supervoxel_index_list, dist_matrix, k):
		'''
		This method finds the k nearest supervoxels to each of the given suporvoxels in supervoxel_list.
		For each supervoxel output is a list of tuples of (dist, i) where dist is the distance between 
		those supervoxles and i is the index of i-th nearest supervoxel.
	
		:param arg1: list of indexes of supervoxels for which the k nearest neighbors are being computed
		:param arg2: distance matrix, n x n matrix, n: number of all the supervoxels in a video
		:param arg3: k -> number of nearest neighbors
		:type arg1: iterable
		:type arg2: 2d array: [][]
		:type arg3: int
		:return: a 2d array where i-th row contains (dist, i)  the distance and indices of k nearest neighbors of supervoxel_list[i] sorted by their  distance
		:rtype: 2d array: [][]
		'''
		assert self.segment is not None, 'Please load a segment `loadSegment(segment_pickle_path)`'
		ans = []
		ap = ans.append
		for  sp_idx in supervoxel_index_list:
			heap = heapq.nsmallest(k,  ((dist, i) for i, dist in enumerate(dist_matrix[sp_idx])))
			heap.sort(key=lambda x: x[0])
			ap(heap)
		return ans

	def averagePrecisionOfSupervoxel(self, supervoxel, ):
		pass
		#neighbors = seg.
	
	
	def __extractFeatures(self, seg):
		'''
		Given an object of the Segmentation class, features of the targets and their neighbors are extracted.

		:return: a dictionary that has the following keys: target, negative, neighbor0, neighbor1, ..., neighbork
			the value of each key is a numpy.array of size n by f, where n is the number of supervoxels in the
			video and f is the size of the feature vector of each supervoxel
		:rtype: dict
		'''
		seg.__class__ = MySegmentation
	#	random_i = 10
	#	from_path = seg.original_path
	#	to_path = '/cs/vml3/mkhodaba/cvpr16/visualization/b1/01/{0:05d}.ppm'
	#	seg.visualizeSegments([seg.supervoxels_list[random_i]], from_path, to_path)
	#	print 'extract features ...'
		features = MySegmentation.getFeatures(seg, 2)
	#	print 'done extracting ...'
		return features

	def __prepare(self, test_model, snapshot_model, features):
		'''
		This function replaces the input of test_model with features from arg3 (features) and 
		replaces the weights of the inner_product_target layer with the ones in snapshot_model.

		:return: None
		'''
		sh = features['target'].shape
		for key in features.keys():
			test_model.blobs[key].reshape(sh[0], sh[1])
		test_model.params['inner_product_target'][0].data[...] = snapshot_model.params['inner_product_target'][0].data[...]
		test_model.params['inner_product_target'][1].data[...] = snapshot_model.params['inner_product_target'][1].data[...]
		test_model.blobs['target'].data[...] = features['target'][...]
	
	def getRepresentations(self, test_model):
		'''
		:return: embedding representation of the input data loaded in the test_model.
		:rtype: numpy.array
		'''
		test_model.forward()
		return test_model.blobs['inner_product_target'].data[...]
	
	def getCosineDistances(self, reps, target):
		dists = np.zeros((reps.shape[0], 2))
		for i in range(reps.shape[0]):
			dists[i][0] = -1*np.dot(reps[i][...], reps[target][...])
			dists[i][1] = i
		#dists= sorted(dists, key=lambda x: x[0])
		return dists

def getEuDistances(reps, target):
	dists = np.zeros((reps.shape[0], 2))
	for i in range(reps.shape[0]):
		dists[i][0] = euclidean(reps[i][...], reps[target][...])
		dists[i][1] = i
	dists= sorted(dists, key=lambda x: x[0])
	return dists


def getCosineDistances_old(reps, target):
	target_s = np.tile(reps[target][...], (reps.shape[0], 1))
	cos_dot = target_s * reps
	dists = np.sum(cos_dot, axis=1)
	dists = np.column_stack((dists, np.arange(reps.shape[0])))
	dists = sorted(dists, key=lambda x: x[0])
	return dists



def visualizeTopN(dists, seg, n):
	'''
	Given distances of supervoxels sp(not given) to all other supervoxels,
	visualizes the n closest supervoxels
	'''
	dd = sorted(dists, key=lambda x: x[0])
	indices = [int(x[1]) for x in dd[1:n]]
	retrieval = [seg.supervoxels_list[i] for i in indices]
	to_path = '/cs/vml3/mkhodaba/cvpr16/visualization/b1/02/{0:05d}.ppm'
	seg.visualizeSegments(retrieval, seg.original_path, to_path)

def visualize(target):
	seg.__class__ = MySegmentation
	from_path = seg.original_path
	to_path = '/cs/vml3/mkhodaba/cvpr16/visualization/b1/01/{0:05d}.ppm'
	seg.visualizeSegments([seg.supervoxels_list[target]], from_path, to_path)
	


def main2():
	em = main()
	seg = em.segments[0]
	fet = extractFeatures(seg)
	prepare(em.test_model, em.model, fet)
	representations = getRepresentations(em.test_model)
	target_id = 150
	cos_dist = getCosineDistances(reps, target_id)
	


def main():
	dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/{name}'
	snapshot_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/snapshot/vml_gpu/many/{name}'	
	net_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/model/{name}'	
	snapshot_name = '_iter_870000.caffemodel'
	net_name = 'model.prototxt'
	embeddingTester = EmbeddingTester(dataset_path.format(name='segmentors_lvl1.p'))
	embeddingTester.loadModel(snapshot_path.format(name=snapshot_name), net_path.format(name=net_name))
	embeddingTester.test_model = caffe.Net(net_path.format(name='model_test.prototxt'), snapshot_path.format(name=snapshot_name), caffe.TEST)

	print 'pickling segments/features ...'
#	segments = pickle.load(open(dataset_path.format(name='segmentors_lvl1.p'), 'r'))
#	features = pickle.load(open(dataset_path.format(name='features_lvl1.p'), 'r'))
#	print 'done pickling ...'

#	embeddingTester.segment = segments[0]
#	embeddingTester.feature = features[0]

#	return embeddingTester
	#for key in features.keys():
	#	embeddingTester.model.blobs[key].reshape(shape[0], shape[1])
	#embeddingTester.model.blobs['target'].data[...] = features['target'][...]
	#embeddingTester.model.forward()
	
	#target_embedding = embedding.model.blobs['inner_product_target']
	
	#return embeddingTetser

#if __name__ == '__main__':
#	main()
	
