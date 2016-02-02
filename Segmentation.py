from scipy.spatial import cKDTree
import numpy as np
import os.path
from Supervoxel import *
from utils import *
from Annotation import *
from enum import Enum
import random
import h5py

class FeatureType(Enum):
	COLOR_HISTOGRAM = 1
	MBH = 2
	CLR_MBH = 3
	CORSO = 4
	#DEEP = 3

class Segmentation(object):
	

	def __init__(self, original_path='./orig/{0:05d}.ppm', segmented_path='./seg/{0:05d}.ppm', annotator=None, segment=None):
		if segment is not None:
			attrs = [a for a in dir(segment) if not a.startswith('__') and not callable(getattr(segment,a))]
			for attr in attrs:
				print attr
				setattr(self, attr, getattr(segment, attr))
			return
		else:
			assert annotator is not None or not isinstance(annotator, Annotator), 'Annotator should be given'
		self.supervoxels = {} # ID -> Supervoxel
		self.frame_to_voxels = {} # frame (int) -> Supervoxel
		self.current_frame = 1
		self.original_path = original_path
		self.segmented_path = segmented_path
		self.__cKDTRee__ = None #cKDTree() for finding the neighbors. This attribute is set in donePrecessing method
		self.supervoxels_list = None # list of all supervoxels. This attribute is set in donePrecessing method
		self.annotator = annotator #an object of the class of Annotator
		self.in_process = False

	#TODO: implement this for faster pickleing
#	def __reduce__(self, path):
#		pass

	def __findLowestThresholdIndex(self,threshold):
		#if self.in_process:
		#	self.doneProcessing()
		assert self.in_process == False, 'processing is not done yet'		
		first = 0
		last = len(self.supervoxels_list)
		while first < last:
			mid = first + (last-first+1)/2
			if self.supervoxels_list[mid].getOverlap() > threshold:
				first = mid				
			else:
				last = mid-1
		return first
		
	def getLabels(self,threshold):
		'''
		:return: an array of length n (number of supervoxels). return[i] is False if 
			 supervoxels_list[i] is background, otherwise True
		:rtype: 1d-array -> []
		'''
		#if self.in_process:
		#	self.doneProcessing()
		assert self.in_process == False, 'processing is not done yet'
		idx = self.__findLowestThresholdIndex(threshold)		
		return [True if i <=idx else False for i in xrange(len(self.supervoxels_list))]

	def getOverlappingSupervoxels(self,threshold):
		#if self.in_process:
		#	self.doneProcessing()
		assert self.in_process == False, 'processing is not done yet'
		#TODO: check if xrange works?
		idx = self.__findLowestThresholdIndex(threshold)
		return range(idx+1)


	
	def addSupervoxels(self, original_img_path, segmented_img_path, frame_number):
		self.in_process = True
		frame_number = frame_number-1
		orig_img = MyImage(original_img_path)
		img = MyImage(segmented_img_path)
		voxel_colors = img.getcolors()
		#print "Colors"
		#for c in voxel_colors:
		#	print c

		self.frame_to_voxels[frame_number] = set()
		for color in voxel_colors:
			if color not in self.supervoxels:
				self.supervoxels[color] = HistogramSupervoxel(color)
			self.frame_to_voxels[frame_number].add(self.supervoxels[color])

		#print img.size
		try:
			labels = self.annotator.labels
		except:
			pass
		for x in range(img.size[0]):
			for y in range(img.size[1]):
				color = img.getpixel(x, y)
				try:
					self.supervoxels[color].addVoxel(x, y, frame_number, orig_img.getpixel(x, y), labels[y][x][frame_number]) 	
				except:
					self.supervoxels[color].addVoxel(x, y, frame_number, orig_img.getpixel(x, y), 0)
		#			print x,y,frame_number
		#			raise
 

	def processNewFrame(self):
		orig_path = self.original_path.format(self.current_frame)
		seg_path = self.segmented_path.format(self.current_frame)
		self.addSupervoxels(orig_path, seg_path, self.current_frame)
		self.current_frame += 1

	#TODO: Re-implement this one!
	#def saveSegments(self, supervoxels_set, from_path='./orig/{0:05d}.ppm', to_path='./save/{0:05d}.ppm'):
	#	mkdirs(to_path)
	#	all_voxels = []
	#	for sv in supervoxels_set:
	#		for p in sv.pixels:
	#			all_voxels.append((p, sv.ID)) #pair of (pixel, ID). ID is usually color in segmented image
	#
	#	all_voxels.sort(key=lambda x: x[0][2]) #sort all pixels of all supervoxels based on frame number
	#
	#	open_frame = all_voxels[0][0][1]
	#	img = MyImage(from_path.format(open_frame))
	#	for pixel, color in all_voxels:
	#		x,y,f = pixel
	#		if open_frame != f:
	#			img.save(to_path.format(open_frame))
	#			open_frame = f
	#			img = MyImage(from_path.format(open_frame))
	#		img.putpixel(x,y, color)		

	def visualizeSegments(self, supervoxels_set, from_path='./orig/{0:05d}.ppm', to_path='./save/{0:05d}.ppm', colors={}):
		mkdirs(to_path)
		all_frames = set()
		default_color = colors.get(-1, None)
		for sv in supervoxels_set:
			all_frames.update(sv.pixels.keys())
			if not sv.ID in colors:
				if default_color:
					colors[sv.ID] = default_color
				else:
					colors[sv.ID] = sv.ID #Assuming that IDs (r,g,b) tuples and unique

		for f in all_frames:
			img = MyImage(from_path.format(f))
			for sv in supervoxels_set:
				clr = colors[sv.ID]
				if f in sv.pixels:
					for x,y in sv.pixels[f]:
						img.putpixel(x,y, clr)
				#		img.putpixel(x,y, sv.ID)
			img.save(to_path.format(f))		

	def doneProcessing(self):
		self.supervoxels_list = self.supervoxels.values()

		#For coros segmentation
		ids = map(lambda x: (x.ID[0]+1)*10**6+(x.ID[1]+1)*10**3+(x.ID[2]+1), self.supervoxels_list)
		self.supervoxels_list_corso = [x for (y, x) in sorted(zip(ids, self.supervoxels_list))]

		self.supervoxels_list.sort(key=lambda sp: sp.overlap_count, reverse=True) #Sort supervoxels_list based of the overlap amount Largest to Lowest
		self.in_process = False

		
		

	def getSupervoxelAt(self, x, y, t):
		pixel = (x,y)
		for sv in self.frame_to_voxels[t]:
			if pixel in sv.pixels[t]:
				return sv
	
	#For Pickling
	def __getstate__(self):
		if hasattr(self, "data"):
			del self.data
		state = {attr:getattr(self,attr) for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self,attr))}
		#state = {'supervoxels': self.supervoxels_list, 'supervoxels2': self.supervoxels_list}		
		return state

	def __setstate__(self, dic):
		for key in dic:
			setattr(self, key, dic[key])

class MySegmentation(Segmentation):
	def __init__(self, original_path='./orig/{0:05d}.ppm', segmented_path='./seg/{0:05d}.ppm', features_path = './features.txt', annotator=None, segment=None):
		if  segment is None:
			#print original_path, segmented_path, len(annotator.labels)
			super(MySegmentation, self).__init__(original_path, segmented_path, annotator)
		else:
			super(Segmentation, self).__init__(segment.original_path, segmented_path, segment)
		self.features_path = features_path

	def getNearestSupervoxelsOf(self, supervoxel, threshold=30):
		pass

	def getKNearestSupervoxelsOf(self, supervoxel, k=6):
		'''
		:param arg1: supervoxel
		:param arg2: number of neighbors
		:type arg1: Supervoxel()
		:type arg2: int
		:return: set of neighbors of supervoxel
		:rtype: set()
		'''
		if not hasattr(self, 'cKDTree'):
			self.__cKDTree__ = cKDTree(np.array([sv.center() for sv in self.supervoxels_list]))
		nearestNeighbors = self.__cKDTree__.query(np.array(supervoxel.center()), k+1)[1] # Added one to the neighbors because the target itself is included
	
		return set(self.supervoxels_list[i] for i in nearestNeighbors[1:])
	
	def prepareData(self, k, number_of_data, feature_vec_size):
		feature_size = feature_vec_size * (1 + k + 1) #One for the target, k for neighbors, one for negative
		data = np.arange(number_of_data*feature_size)
		#data = data.reshape(number_of_data, 1, 1, feature_size)
		data = data.reshape(number_of_data, feature_size)
		data = data.astype('float32')

		print "data.shape: ", data.shape
		return data

	def dummyData(self, number_of_data, feature_vec_size):
		'''
		:param arg1: number of data (n)
		:param arg2: length of the feature vector of each supervoxel (k)
		:type arg1: int
		:type arg2: int
		:return: an array of size n by k
		:rtype: numpy.array
		'''
		#TODO: what the hell? just use np.zeros(n,f)
		data = np.arange(number_of_data*feature_vec_size)
		data = data.reshape(number_of_data, feature_vec_size)
		data = data.astype('float32')

		return data

	def setFeatureType(self, feature_type):
		self.feature_type = feature_type
	
#	def _extract_features_from_supervoxel_(self, sv):
#		if self.feature_type == FeatureType.COLOR_HISTOGRAM:
#			return sv.getFeature()
#		elif self.feature_type == FeatureType.MBH:
#		else:
#			raise "Feature type is wrong!"
		
	def _extract_color_histogram(self, k, negative_numbers):
		assert k >= 2, 'K < 2: At least 2 neighbors is needed'

		supervoxels = set(self.supervoxels_list)
		feature_len = len(self.supervoxels_list[0].getFeature())
		n = len(supervoxels) * negative_numbers
		data = {'target':self.dummyData(n, feature_len), 'negative':self.dummyData(n, feature_len)}
		for i in range(k):
			data['neighbor{0}'.format(i)] = self.dummyData(n, feature_len)
		for i, sv in enumerate(self.supervoxels_list):
			neighbors = self.getKNearestSupervoxelsOf(sv, k) 
			#print 'neighbors', len(neighbors)
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negatives = random.sample(supervoxels, negative_numbers) #Sample one supervoxel as negative
			#neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)
			idx = i*negative_numbers
			data['target'][idx][...] = sv.getFeature()
			for j, nei in enumerate(neighbors):
				data['neighbor{0}'.format(j)][idx][...] = nei.getFeature()
				#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data['negative'][idx][...] = negatives[0].getFeature()
			for neg in xrange(1, negative_numbers):
				idx = i*negative_numbers+neg
				data['target'][idx][...] = data['target'][idx][...]
				for j, nei in enumerate(neighbors):
					data['neighbor{0}'.format(j)][idx][...] = data['neighbor{0}'.format(j)][idx][...]
					#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
				data['negative'][idx][...] = negatives[neg].getFeature()
								

		#print data.keys()
		return data

	def _read_features(self):
		feature_len = 192
		features = np.zeros((len(self.supervoxels_list) ,feature_len))
		sv2id = {sv.ID:i for i,sv in enumerate(self.supervoxels_list)}
		print 'len(supervoxels) = %d' % features.shape[0]
		i = 0
		with open(self.features_path, 'r') as f:
			for num_line, l in enumerate(f):
				pass
		num_line+=1
		print "Number of lines", num_line
		print (14,106,23) in sv2id
		print (23,106,14) in sv2id
		with open (self.features_path, 'r') as reader:
			for i in xrange(num_line-1):
				line = reader.next()
				line = line.split()
				sv_id = (int(line[2]), int(line[1]), int(line[0]))
				f = np.array(map(float, line[3:]))
				assert f.shape[0]%feature_len == 0, 'feature len(%d) is not disiable by %d' % (f.shape[0], feature_len) 
				f = f.reshape((f.shape[0]/feature_len, feature_len))
				f = np.mean(f, 0)
				assert sv_id in sv2id, 'sv_id(%d,%d,%d) not in sv2id, i=%d' % (sv_id[2], sv_id[1], sv_id[0], i)
				idx = sv2id[sv_id]			
				features[idx][...] = f[...]
				i+=1
		return features
	
	def _extract_mbh(self, k, negative_numbers):
		if hasattr(self, "data"):
			return self.data
	
		sv2id = {sv.ID:i for i,sv in enumerate(self.supervoxels_list)}
		features= self._read_features()		
		feature_len = features.shape[1] #first three numbers are the id
		supervoxels = set(self.supervoxels_list)
		n = len(supervoxels) * negative_numbers
		data = {'target':self.dummyData(n, feature_len), 'negative':self.dummyData(n, feature_len)}
		for i in range(k):
			data['neighbor{0}'.format(i)] = self.dummyData(n, feature_len)
		for i, sv in enumerate(self.supervoxels_list):
			neighbors = self.getKNearestSupervoxelsOf(sv, k) 
			#print 'neighbors', len(neighbors)
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negatives = random.sample(supervoxels, negative_numbers) #Sample one supervoxel as negative
			#neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)
			idx = i*negative_numbers
			data['target'][idx][...] = features[sv2id[sv.ID]][...]#sv.getFeature()
			for j, nei in enumerate(neighbors):
				data['neighbor{0}'.format(j)][idx][...] = features[sv2id[nei.ID]][...]#nei.getFeature()
				#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data['negative'][idx][...] = features[sv2id[negatives[0].ID]][...]#negatives[0].getFeature()
			for neg in xrange(1, negative_numbers):
				idx = i*negative_numbers+neg
				data['target'][idx][...] = data['target'][idx][...]
				for j, nei in enumerate(neighbors):
					data['neighbor{0}'.format(j)][idx][...] = data['neighbor{0}'.format(j)][idx][...]
					#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
				data['negative'][idx][...] = features[sv2id[neg.ID]][...]#negatives[neg].getFeature()
								

			#print data.keys()
		self.data = data
		return data

	def _extract_clr_mbh(self, k, negative_numbers):
		if hasattr(self, "data"):
			return self.data
	
		sv2id = {sv.ID:i for i,sv in enumerate(self.supervoxels_list)}
		feature_len1 = len(self.supervoxels_list[0].getFeature())
		features= self._read_features()		
		feature_len = features.shape[1]+feature_len1 #first three numbers are the id
		supervoxels = set(self.supervoxels_list)
		n = len(supervoxels) * negative_numbers
		data = {'target':self.dummyData(n, feature_len), 'negative':self.dummyData(n, feature_len)}
		for i in range(k):
			data['neighbor{0}'.format(i)] = self.dummyData(n, feature_len)
		for i, sv in enumerate(self.supervoxels_list):
			neighbors = self.getKNearestSupervoxelsOf(sv, k) 
			#print 'neighbors', len(neighbors)
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negatives = random.sample(supervoxels, negative_numbers) #Sample one supervoxel as negative
			#neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)
			idx = i*negative_numbers
			data['target'][idx][...] = np.append(features[sv2id[sv.ID]][...], sv.getFeature())#sv.getFeature()
			for j, nei in enumerate(neighbors):
				data['neighbor{0}'.format(j)][idx][...] = np.append(features[sv2id[nei.ID]][...], nei.getFeature())#nei.getFeature()
				#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data['negative'][idx][...] = np.append(features[sv2id[negatives[0].ID]][...], negatives[0].getFeature())#negatives[0].getFeature()
			for neg in xrange(1, negative_numbers):
				idx = i*negative_numbers+neg
				data['target'][idx][...] = data['target'][idx][...]
				for j, nei in enumerate(neighbors):
					data['neighbor{0}'.format(j)][idx][...] = data['neighbor{0}'.format(j)][idx][...]
					#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
				data['negative'][idx][...] = np.append(features[sv2id[neg.ID]][...], negatives[neg].getFeature())#negatives[neg].getFeature()
								

			#print data.keys()
		self.data = data
		return data

	def _read_corso_features(self):
		
		features = h5py.File(self.features_path,'r')
		return np.array(features['hist']).T

	def _extract_corso(self, k, negative_numbers):
		if hasattr(self, "data_corso"):
			return self.data_corso
	
		sv2id = {sv.ID:i for i,sv in enumerate(self.supervoxels_list_corso)}
		features= self._read_corso_features()	
		print features
		assert features.shape[1] == 42, 'features size is wrong'	
		feature_len = features.shape[1]
		supervoxels = set(self.supervoxels_list_corso)
		n = len(supervoxels) * negative_numbers
		data = {'target':self.dummyData(n, feature_len), 'negative':self.dummyData(n, feature_len)}
		for i in range(k):
			data['neighbor{0}'.format(i)] = self.dummyData(n, feature_len)
		for i, sv in enumerate(self.supervoxels_list_corso):
			neighbors = self.getKNearestSupervoxelsOf(sv, k) 
			#print 'neighbors', len(neighbors)
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negatives = random.sample(supervoxels, negative_numbers) #Sample one supervoxel as negative
			#neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)
			idx = i*negative_numbers
			data['target'][idx][...] = features[sv2id[sv.ID]][...]
			for j, nei in enumerate(neighbors):
				data['neighbor{0}'.format(j)][idx][...] = features[sv2id[nei.ID]][...]
				#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data['negative'][idx][...] = features[sv2id[negatives[0].ID]][...]

			for neg in xrange(1, negative_numbers):
				idx = i*negative_numbers+neg
				data['target'][idx][...] = data['target'][idx][...]
				for j, nei in enumerate(neighbors):
					data['neighbor{0}'.format(j)][idx][...] = data['neighbor{0}'.format(j)][idx][...]
					#data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
				data['negative'][idx][...] = features[sv2id[neg.ID]][...]
								

			#print data.keys()
		self.data_corso = data
		return data
		
	def getFeatures(self, k, negative_numbers=1, feature_type=FeatureType.COLOR_HISTOGRAM):
		'''
		:param arg1: number of nieghbors (k)
		:type arg1: int
		:return: a dictionary that has the following keys: target, negative, neighbor0, neighbor1, ..., neighbork
			the value of each key is a numpy.array of size n by f, where n is the number of supervoxels in the
			video and f is the size of the feature vector of each supervoxel
		:rtype: dict
		
		'''
		assert k >= 2, 'K < 2: At least 2 neighbors is needed'
		if feature_type == FeatureType.COLOR_HISTOGRAM:
			return self._extract_color_histogram(k, negative_numbers)
		elif feature_type == FeatureType.MBH:
			return self._extract_mbh(k, negative_numbers)
		elif feature_type == FeatureType.CLR_MBH:
			return self._extract_clr_mbh(k, negative_numbers)
		elif feature_type == FeatureType.CORSO:
			return self._extract_corso(k, negative_numbers)
		else:
			raise "Feature type is invalid"


	def getFeaturesInOne(self):
		supervoxels = set(self.supervoxels_list)
		k = 1
		feature_len = len(self.supervoxels_list[0].getFeature())
		data = self.prepareData(k, len(supervoxels), feature_len)

		for i,sv in enumerate(self.supervoxels_list):
			neighbors = self.getKNearestSupervoxelsOf(sv, k)
			supervoxels.difference_update(neighbors) #ALl other supervoxels except Target and its neighbors
			#TODO: Implement Hard negatives. Maybe among neighbors of the neighbors?
			# Or maybe ask for K+n neighbors and the last n ones could be candidate for hard negatives
			negative = random.sample(supervoxels, 1)[0] #Sample one supervoxel as negatie
			neighbors.remove(sv)

			#when everything is done we put back neighbors to the set
			supervoxels.update(neighbors)
			supervoxels.add(sv)

			data[i][0:feature_len] = sv.getFeature()
			for j, nei in enumerate(neighbors):
				data[i][(j+1)*feature_len:(j+2)*feature_len] = nei.getFeature()
			data[i][(k+1)*feature_len:(k+2)*feature_len] = negative.getFeature()
		return data

class MyMotionSegmentation(MySegmentation):
	def __init__(self, original_path='./orig/{0:05d}.ppm', segmented_path='./seg/{0:05d}.ppm', annotator=None, segment=None):
		if  segment is None:
			print original_path, segmented_path, len(annotator.labels)
			super(MySegmentation, self).__init__(original_path, segmented_path, annotator)
		else:
			super(Segmentation, self).__init__(segment.original_path, segmented_path, segment)

	



class DB:
	
	def __init__(self, path):
		self.path = path
		self.h5pyDB = h5py.File(path, 'w')

	def __db__(self):
		return self.h5pyDB

	def save(self, data, name='data'):
		if isinstance(data, dict):
			for name, dataset in data.iteritems():
				self.h5pyDB.create_dataset(name, data=dataset, compression='gzip', compression_opts=1)
	
		else:
			data = np.array(data)
			data = data.astype('float32')
			self.h5pyDB.create_dataset(name, data=data, compression='gzip', compression_opts=1)
	
	
	def close(self):
		self.h5pyDB.close()




def doSegmentation(**kargs):
	pass
def doDataCollection(**kargs):
	pass


import cPickle as pickle
from Annotation import JHMDBAnnotator as JA
def main():

	frame_format = '{0:05d}.ppm'
	seg_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/seg/{1:02d}/' #+ frame_format
	orig_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/' #+ frame_format
	first_output = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/mymethod/{1:02d}/first/'#.format(level)	
	output_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b{0}/mymethod/{1:02d}/output/'#.format(level)
	dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/{name}'
	annotation_path = '/cs/vml3/mkhodaba/cvpr16/dataset/{name}_mask/mask.csv'
	# Preparing data for 
	#segmentor = Segmentation(orig_path, seg_path+frame_format)
	level = 1
	segmentors = []
	vid_num = 4
	frames_per_video = 31
	for d in range(1,vid_num):
		print 'b{0}'.format(d)
		annotator = JA(annotation_path.format(name='b'+str(d)))
		segmentor = MySegmentation(orig_path.format(d)+frame_format, seg_path.format(d,level)+frame_format, annotator)
		for i in range(1, frames_per_video):
			print "processing frame {i}".format(i=i)
			segmentor.processNewFrame()
		segmentor.doneProcessing()
		segmentors.append(segmentor)
		print "Total number of supervoxels: {0}".format(len(segmentor.supervoxels))
		print
		
	#sv = segmentor.getSupervoxelAt(27, 127, 20)
	#print sv
	#supervoxels = segmentor.getKNearestSupervoxelsOf(sv, 6)
	#supervoxels.remove(sv)
	#for s in supervoxels:
	#	print s


	#TODO check if features are correct
	##for sv in segmentor.supervoxels_list:
		##print sv.getFeature()
		##print "ID: {0}".format(sv.ID)		

		#R_hist = [0 for i in xrange(13)]
		#G_hist = [0 for i in xrange(13)]
		#B_hist = [0 for i in xrange(13)]
		#R_hist[int(sv.ID[0]/20)] += 1
		#G_hist[int(sv.ID[1]/20)] += 1
		#B_hist[int(sv.ID[2]/20)] += 1
		#print R_hist+G_hist+B_hist
		#print sum(sv.getFeature())/3		
		#print "Num pixels: {0}".format(sv.number_of_pixels)

	pickle.dump(segmentors[0], open(dataset_path.format(name='segment_1.p'), 'w'))
	pickle.dump(segmentors[1], open(dataset_path.format(name='segment_2.p'), 'w'))
	pickle.dump(segmentors[2], open(dataset_path.format(name='segment_3.p'), 'w'))

	'''
	#TODO create database
	mkdirs(dataset_path)
	database = DB(dataset_path.format(name='b1b2_train_16bins_lvl{0}.h5'.format(level)))

	print 'Collecting features ...'
	neighbor_num = 6
	keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]	
	features = segmentors[0].getFeatures(neighbor_num)
	print 'shape features', features['target'].shape
	feats = [features]
	print 'video 1 done!'
	for i in range(1, len(segmentors)-1):
		tmp = segmentors[i].getFeatures(neighbor_num)
		feats.append(tmp)
		for key in keys:
			features[key] = np.append(features[key], tmp[key], axis=0)	
		print 'video {0} done!'.format(i+1)
	#print data
	#database_path = '
	print 'saving to database ...'
	for name, data in features.iteritems():
		database.save(data, name)
	#database.save(dataset)	
	database.close()


	database = DB(dataset_path.format(name='b3_test_16bins_lvl{0}.h5'.format(level)))

	print 'Collecting features ...'
	neighbor_num = 6
	features = segmentors[-1].getFeatures(neighbor_num)
	print 'shape features', features['target'].shape
	feats = [features]
	print 'video 3 done!'
	#print data
	#database_path = '
	print 'saving to database ...'
	for name, data in features.iteritems():
		database.save(data, name)
	#database.save(dataset)	
	database.close()


	'''
	print 'done!'

	#for i in range(len(segmentors)):
	#	print i
	#	segmentors[i] = Segmentation(segment=segmentors[i])

	#print 'pickle segments ...'
	#pickle.dump( segmentors, open(dataset_path.format(name='segmentors_lvl1.p'), 'w'))
	#print 'pickle features ...'	
	#pickle.dump( feats, open(dataset_path.format(name='features_lvl1.p'), 'w'))



if __name__ == "__main__":
	main()





