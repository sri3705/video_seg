from Embedding import *
from Annotation import JHMDBAnnotator as JA, Evaluator as EVAL
import numpy as np
import random
import time
from configs import getConfigs

conf = getConfigs(43)
snapshot_path = conf.solver['snapshot_prefix']
db = conf.db_settings[conf.db]
action = 'pour'
video = db['video_name'][action][0]
level = db['level']
dataset_path = db['pickle_path'].format(action_name='pour', video_name=video, level=level)
#dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/{name}'
net_path = conf.model['model_prototxt_path']
#net_path = {name}'	
annotation_path = db['annotation_path'].format(action_name=action, video_name=video)
#annotation_path = '/cs/vml3/mkhodaba/cvpr16/dataset/{name}'
#snapshot_name = '_iter_870000.caffemodel'
snapshot_name = '_iter_25000.caffemodel'

#model_name = 'model.prototxt'
test_model_path = '/cs/vml2/mkhodaba/cvpr16/expriments/8/test.prototxt'
snapshot_path = conf.solver['snapshot_prefix']#'/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/snapshot/vml_gpu/256bins/'	

print net_path

print dataset_path
print 

start = time.time()
print "loading segments ..."
segment = pickle.load(open(dataset_path,'r'))
end = time.time()

print
print "Done. Elapsed time: {0}".format(end-start)
start = end
print 'Computing embeddings'
embeddingTester = EmbeddingTester(segment)
embeddingTester.loadModel(snapshot_path+snapshot_name, net_path, test_model_path)

#annot = JA(annotation_path.format(name='/b1_mask/mask.csv')

end = time.time()
print
print "Done. Elapsed time: {0}".format(time.time()-start)
start = end
print "Computing distance matrix"

distances = embeddingTester.computeDistanceMatrix()
distances2 = embeddingTester.computeBaselineDistanceMatrix()

print "elappsed time: {0}".format(time.time()-end)


def meanAP(retrieves, labels):
	aps = []
	for rets in retrieves:
		ap = EVAL.AP(rets, labels)
		if np.isnan(ap):
			ap = 0
		aps.append(ap)
	return aps
threshold = 0.5

def print_map(embeddingTester, threshold, k):
	positive_supervoxels = embeddingTester.segment.getOverlappingSupervoxels(threshold)
	labels = embeddingTester.segment.getLabels(threshold)
	samples = random.sample(positive_supervoxels, len(positive_supervoxels))
	print 'Threshold = ', threshold
	retrieves = embeddingTester.retrieveKNearest(samples, distances, k)
	retrieves2 = embeddingTester.retrieveKNearest(samples, distances2, k)
	ours = meanAP(retrieves, labels)
	baseline = meanAP(retrieves2, labels)
	ours_map = sum(ours)/len(ours)
	baseline_map = sum(baseline)/len(baseline)
	print '\tK = ', k
	print '\t   Ours:     ', ours_map
	print '\t   Baseline: ', baseline_map

print_map(embeddingTester, threshold, 200)

res = {}
if 1 == 2:
	for th in xrange(5, 6, 1):
		threshold = th*0.1
		positive_supervoxels = embeddingTester.segment.getOverlappingSupervoxels(threshold)
		labels = embeddingTester.segment.getLabels(threshold)
		samples = random.sample(positive_supervoxels, len(positive_supervoxels))
		print 'Threshold = ', threshold
		res[threshold] = []
		maxx = -2000
		for k in range(100, 120, 5):
			retrieves = embeddingTester.retrieveKNearest(samples, distances, k)
			retrieves2 = embeddingTester.retrieveKNearest(samples, distances2, k)

			ours = meanAP(retrieves, labels)
			baseline = meanAP(retrieves2, labels)
			ours_map = sum(ours)/len(ours)
			baseline_map = sum(baseline)/len(baseline)
			if maxx < ours_map-baseline_map:
				max_k = k
				max_ours = ours_map
				max_baseline = baseline_map
				maxx = ours_map-baseline_map
	#		res[threshold].append((k, average_precisions, baseline_ap))
			print '\tK = ', k
			print '\t   Ours:     ', ours_map
			print '\t   Baseline: ', baseline_map

	#		print 'k = {0}'.format(k)
	#		print 'meanAP ours', sum(average_precisions)/len(average_precisions)
	#		print 'meanAP baseline', sum(baseline_ap)/len(baseline_ap)
		print '*********************************'	
		print 'Threshold = ', threshold
		print '\tK = ', max_k
		print '\t  Ours:     ', max_ours
		print '\t  Baseline: ', max_baseline

	for threshold, vals in res.iteritems():
		print 'Threshold = ', threshold
		v = max(vals, key=lambda x: x[1]-x[2])
		print '\tK =', v[0]
		print '\t   Ours:     ', v[1]
		print '\t   Baseline: ', v[2]

threshold = 0.5
positive_supervoxels = embeddingTester.segment.getOverlappingSupervoxels(threshold)
labels = embeddingTester.segment.getLabels(threshold)

def doROC(idx, labels, distances, arg=30, k=200, clr="r"):	
	rets = embeddingTester.retrieveKNearest([idx], distances, k)[0]
	score = [dist for dist, i in rets]
	y = [1 if labels[i] else 0 for dist, i in rets]
	return EVAL.ROCCurve(score, y, arg, clr)

def visualizeNearest(sp_idx, embeddingTester, distances, k, colors=None):
	rets = embeddingTester.retrieveKNearest([sp_idx], distances, k)[0]
	poss = [embeddingTester.segment.supervoxels_list[i[1]] for i in rets]
	original_path = '/cs/vml3/mkhodaba/cvpr16/dataset/b1/{0:05d}.ppm'
	to_path = '/cs/vml3/mkhodaba/cvpr16/dataset/test_b1/{0:05d}.ppm'
	embeddingTester.segment.visualizeSegments([embeddingTester.segment.supervoxels_list[sp_idx]], original_path, to_path, colors)
	to_path = '/cs/vml3/mkhodaba/cvpr16/dataset/test_b1_neighbors/{0:05d}.ppm'
	mkdirs('/cs/vml3/mkhodaba/cvpr16/dataset/test_b1_neighbors/')
	segment.visualizeSegments(poss, original_path, to_path, colors)

#visualizeNearest(positive_suporvoxels[10], embeddingTester, distances2)


