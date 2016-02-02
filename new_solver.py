import caffe
from numpy import zeros
import numpy as np
import cPickle as pickle
from configs import getConfigs
from Segmentation import *
import heapq
from Annotation import Evaluator as EVAL
from Logger import *
import sys
from scipy.io import savemat, loadmat

def meanAP(retrieves, labels):
	aps = []
	for rets in retrieves:
		ap = EVAL.AP(rets, labels)
		if np.isnan(ap):
			ap = 0
		aps.append(ap)
	return aps

def retrieveKNearest(supervoxel_index_list, dist_matrix, k):
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
	ans = []
	ap = ans.append
	for  sp_idx in supervoxel_index_list:
		heap = heapq.nsmallest(k,  ((dist, i) for i, dist in enumerate(dist_matrix[sp_idx])))
		heap.sort(key=lambda x: x[0])
		ap(heap)
	return ans

def print_map(segment, threshold, k, distances2, distances, logger):
	positive_supervoxels = segment.getOverlappingSupervoxels(threshold)
	labels = segment.getLabels(threshold)
	samples = random.sample(positive_supervoxels, len(positive_supervoxels))
	logger.log( 'Threshold = {0}'.format(threshold))
	retrieves = retrieveKNearest(samples, distances, k)
	retrieves2 = retrieveKNearest(samples, distances2, k)
	ours = meanAP(retrieves, labels)
	baseline = meanAP(retrieves2, labels)
	ours_map = sum(ours)/len(ours)
	baseline_map = sum(baseline)/len(baseline)
	logger.log('\tK = {0}'.format(k))
	logger.log('\t   Ours:     {0}'.format(ours_map))
	logger.log('\t   Baseline: {0}'.format(baseline_map))

def getRepresentations(solver, segment, idx):
	seg_size = len(segment.supervoxels_list)
	print "number of test nets:", len(solver.test_nets)
	assert idx < len(solver.test_nets), "number of test nets({0}) is less than idx({1})".format(len(solver.test_nets), idx)
	data = solver.test_nets[idx].blobs['inner_product_target'].data
	assert data.shape[0] == 1, 'batch size != ? ... this assert is not important'
	feature_len = data.shape[1]
	reps = np.zeros((seg_size, feature_len))
	for i in xrange(seg_size):
		solver.test_nets[idx].forward()
		reps[i][...] = solver.test_nets[idx].blobs['inner_product_target'].data[...]
	return reps

def getVSB100Representation(conf, solver):
	video_info_path = db_settings['video_info_path'] #'/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/{action_name}_vidinfo.mat'
	video_info = loadmat(video_info_path) #video_info = [mapped, labelledlevelvideo, numberofsuperpixelsperframe]
	framebelong = video_info['framebelong']
	superpixels_num = len(framebelong)
	net = solver.test_nets[0]
	data = net.blobs['inner_product_target'].data
	assert data.shape[0] == 1, 'batch size != ? ... this assert is not important'
	feature_len = data.shape[1]
	reps = np.zeros((superpixels_num, feature_len))
	for i in xrange(superpixels_num):
		net.forward()
		reps[i][...] = net.blobs['inner_product_target'].data[...]
	return reps

def computeDistanceMatrix(representations):
	return -1*(representations.dot(representations.T))

def getBaselineRepresentations(segment, k, feature_type):
	segment.__class__ = MySegmentation
	features = MySegmentation.getFeatures(segment,k, feature_type=feature_type)
	return features

if __name__ == '__main__':
	print "START"
	caffe.set_mode_gpu()
	#caffe.set_decive(2)
	if len(sys.argv) == 1:
		conf = getConfigs(-1)
	else:
		conf = getConfigs(int(sys.argv[1]))

	logger = Logger(LogType.FILE, conf.solver['_solver_log_path'])
	threshold = 0.6

	db = conf.db_settings
	model_prototxt_path = conf.model['model_prototxt_path']
	solver_prototxt_path = conf.solver['_solver_prototxt_path']
	test_interval = conf.solver['test_interval'] #10000
	niter = conf.solver['max_iter'] #max(100000, conf.solver['max_iter']) #500000
	train_interval = conf.solver['_train_interval'] #1000
	termination_threshold = conf.solver['_termination_threshold']
	print "net"
	net = caffe.Net(model_prototxt_path, caffe.TRAIN)
	#net.set_device(2)
	#net.set_mode_gpu()
	#solver = caffe.SGDSolver(root+'solver.prototxt')
	print "solver", solver_prototxt_path
	solver = caffe.SGDSolver(solver_prototxt_path)

	if conf.db == 'jhmdb':
		action = db['action_name'][0]
		print db['video_name'][action]
		level = db['level']
		segments = []
		baseline_distance = []
		test_test = False
		if test_test:
			for video in db['video_name'][action]:
				dataset_path = db['pickle_path'].format(action_name='pour', video_name=video, level=level)
				segment = pickle.load(open(dataset_path,'r'))
				segments.append(segment)
				print video
				baseline_distance.append(computeDistanceMatrix(getBaselineRepresentations(segment, conf.model['number_of_neighbors'], conf.model['feature_type'])['target']))
		segment_idx = 0


	# losses will also be stored in the log
	train_loss = np.array([])
	test_acc = zeros(int(np.ceil(niter / test_interval)))
	test_loss = 0
	# the main solver loop
	it = -1
	prev_loss = 100000000
	diff_loss = 100
	min_iter = 5002
	print "--------------------------------------"
	print "Configs for experiment:",conf.experiment_number
	print "--------------------------------------"
	while (it < niter and abs(diff_loss) >= termination_threshold) or it < min_iter:
		it += 1
		solver.step(1)  # SGD by Caffe

		#print solver.net.blobs['Tile1'].data.shape
		#exit()
		train_loss= np.append(train_loss, solver.net.blobs['loss'].data)
		if it % train_interval == 0:
			logger.log( 'Iteration'+str( it)+ '...' )
			current_loss = np.mean(train_loss[-train_interval:])
			prev_loss, diff_loss = current_loss, prev_loss-current_loss
			logger.log( 'Average Train Loss [last 1000]: {0} -- Train Loss Std:{1}'.format(current_loss, np.std(train_loss[-train_interval:])))
			logger.log('Minimum Train Loss [last 1000]:{0}'.format(np.amin(train_loss[-train_interval:])))
			logger.log('Improvement [last 1000]: {0}'.format(diff_loss))

		#TODO TESTING!!!!
		'''
		if it % test_interval == 0:
			if test_test:
				logger.log("========================TESTING TIME========================")
				for segment_idx in xrange(len(segments)):
					segment = segments[segment_idx]
					reps = getRepresentations(solver, segment, segment_idx)
					cosine_distances = computeDistanceMatrix(reps)
					print len(cosine_distances), len(baseline_distance[segment_idx]), len(segment.supervoxels_list)
					#assert cosine_distances.shape == baseline_distance.shape, "distance matrices size doesn't match"
					print_map(segment, threshold, 200, baseline_distance[segment_idx], cosine_distances, logger)
					logger.log("-"*100)
				logger.log("========================DONE TESTING========================")
			#TODO compute distance
			#TODO mAP
		#	print 'Test Loss:', solver.test_nets[0].blobs['loss'].data
		#	test_loss = solver.test_nets[0].blobs['loss'].data
		'''
	if conf.db == 'jhmdb':
		dataset_path = db['pickle_path'].format(action_name='dadada', video_name='baba', level=1)
		segment = pickle.load(open(dataset_path,'r'))
		reps = getRepresentations(solver, segment, segment_idx)
		cosine_distances = computeDistanceMatrix(reps)
		dic = {'chi_dist': cosine_distances}
		savemat('/cs/vml3/mkhodaba/cvpr16/code/lu/ActionSeg/data/demo/feature/Diving-Side_03/chi_dist_'+str(conf.experiment_number)+'.mat', dic)
		savemat('/cs/vml3/mkhodaba/cvpr16/code/lu/ActionSeg/data/demo/feature/Diving-Side_03/deep_features_'+str(conf.experiment_number)+'.mat', {'deepf':reps})
		if 1 == 2:
			logger.log("======================== Final Testing ========================")
			for video in db['video_name'][action]:
				dataset_path = db['pickle_path'].format(action_name='pour', video_name=video, level=level)
				segment = pickle.load(open(dataset_path,'r'))
				segments.append(segment)
				print video
				baseline_distance.append(computeDistanceMatrix(getBaselineRepresentations(segment, conf.model['number_of_neighbors'], conf.model['feature_type'])['target']))

			logger.log("There are {0} videos.".format(len(segments)))
			for segment_idx in xrange(len(segments)):
				segment = segments[segment_idx]
				reps = getRepresentations(solver, segment, segment_idx)
				cosine_distances = computeDistanceMatrix(reps)
				print len(cosine_distances), len(baseline_distance[segment_idx]), len(segment.supervoxels_list)
				#assert cosine_distances.shape == baseline_distance.shape, "distance matrices size doesn't match"
				print_map(segment, threshold, 200, baseline_distance[segment_idx], cosine_distances, logger)
				logger.log("-------------------------------------------------------------")
			logger.log("============================= DONE ============================")
	logger.close()
	print "Experiment done!"

	#np.savetxt(root+'embedding.txt', solver.net.params['inner_product_target'][0].data)
	#with open(root+'embedding.txt', 'w') as f:
	#	f.write(str(solver.net.params['inner_product_target'][0].data))
	#print solver.net.params['inner_product_negative'][0].data

	#print solver.net.params['inner_product_target'][0].data


	    # store the output on the first test batch
	    # (start the forward pass at conv1 to avoid loading new data)
	    #solver.test_nets[0].forward(start='conv1')
	    #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

	    # run a full test every so often
	    # (Caffe can also do this for us and write to a log, but we show here
	    #  how to do it directly in Python, where more complicated things are easier.)
	    #if it % test_interval == 0:
	    #   print 'Iteration', it, 'testing...'
	    #    correct = 0
	    #    for test_it in range(100):
	    #        solver.test_nets[0].forward()
	    #        correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
	    #                       == solver.test_nets[0].blobs['label'].data)
	    #    test_acc[it // test_interval] = correct / 1e4
