#In the name of God

from Supervoxel import *

dataset_path = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/dataset/{name}'
segmentors = pickle.load( segmentors, open(dataset_path.format(name='segmentors.p'), 'r'))


