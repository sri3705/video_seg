#In the name of GOD


import cPickle as pickle
from Segmentation import *
from Annotation import JHMDBAnnotator as JA
import time
from configs import getConfigs

def create_dbs():
	configs = getConfigs()
	frame_format = configs.frame_format
	seg_path = configs.seg_path
	orig_path = configs.orig_path
	first_output = configs.first_output
	output_path = configs.output_path
	dataset_path = configs.dataset_path
	annotation_path = configs.annotation_path

	feature_name = '256bin'
	level = 2
	segmentors = []
	vid_num = 2
	frames_per_video = 31
	if 1 == 1:
		for dd in range(vid_num):
			d = dd+1
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
	
		try:
			mkdirs(dataset_path)
		except:
			pass
		print 'Piclking ...'
		t = time.time()
		for i in range(vid_num):
			pickle.dump(segmentors[i], open(dataset_path.format(name='segment_{0}.p'.format(i+1)), 'w'))
			print '{0}-th done. time elapsed: {1}'.format(i+1, time.time()-t)
			t = time.time()

		#TODO create database
	else:
		for i in range(vid_num):
			segmentors.append(pickle.load(open(dataset_path.format(name='segment_{0}.p'.format(i+1)), 'r')))
		

	database = DB(dataset_path.format(name='videos{v}_feature{f}_lvl{l}.h5'.format(\
							v='_'.join(map(str,range(1,vid_num))),
							f=feature_name,
							l=level)))
		
	print 'Collecting features ...'
	neighbor_num = 6
	keys = ['target', 'negative'] + [ 'neighbor{0}'.format(i) for i in range(neighbor_num)]	
	features = segmentors[0].getFeatures(neighbor_num)
	print 'shape features', features['target'].shape
	feats = [features]
	print 'video 1 done!'
	for i in range(1, len(segmentors)):
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


	print 'done!'

	#for i in range(len(segmentors)):
	#	print i
	#	segmentors[i] = Segmentation(segment=segmentors[i])

	#print 'pickle segments ...'
	#pickle.dump( segmentors, open(dataset_path.format(name='segmentors_lvl1.p'), 'w'))
	#print 'pickle features ...'	
	#pickle.dump( feats, open(dataset_path.format(name='features_lvl1.p'), 'w'))

if __name__ == '__main__':
	create_dbs()
