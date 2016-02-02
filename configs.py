#In the name of GOD
from utils import *
from Logger import *
import cPickle as pickle
from Segmentation import *

class Config:
    def __init__(self, experiment_number=None):
        self.experiments_root = '/cs/vml2/smuralid/projects/eccv16/python/experiments/'
        if not experiment_number:
            self.__create_config__()

    def __create_config__(self):
        self.frame_format = '{0:05d}.png'
        self.experiment_number = max([0]+map(int, getDirs(self.experiments_root)))+1
        self.experiments_path = self.experiments_root+'/{0}/'.format(self.experiment_number)
        mkdirs(self.experiments_path)
        self.log_path = self.experiments_path+'/log.txt'
        self.log_type = LogType.FILE
        self.db = 'vsb100' # self.db = 'jhmdb' or 'vsb100'
        # self.db = 'jhmdb'
        self.model = {
            'batch_size':		256,
            'number_of_neighbors':	8, #number of neighbors around the target superpixel
            'inner_product_output':	1024, #2*(3*256+192),
            'weight_lr_mult':	1,
            'weight_decay_mult':	1,
            'b_lr_mult':		2,
            'b_decay_mult':		0,
            'model_prototxt_path':	self.experiments_path+'/model.prototxt',
            'test_prototxt_path':	self.experiments_path+'/test.prototxt',
            'database_list_path':	self.experiments_path+'/database_list.txt',
            'feature_type':		FeatureType.CORSO,#FeatureType.COLOR_HISTOGRAM#
        }

        self.solver = {
            'weight_decay':		0.0001,
            'base_lr':		    0.001,
            'momentum': 		0.9,
            'gamma':		    0.4,
            'power':		    0.75,
            'display':		    50,
            'test_interval':	50,
            'test_iter':		1,
            'snapshot':		5000,
            'lr_policy': 		"step",
            'stepsize':		50,
            'snapshot_prefix':	self.experiments_path+'/snapshot/',
            'net':			self.model['test_prototxt_path'],
            '_train_net':		self.model['model_prototxt_path'],
            '_test_nets':		self.model['test_prototxt_path'],
            'max_iter':		100000,
            '_train_interval':	50,
            '_termination_threshold':0.0004,
            '_solver_prototxt_path':	self.experiments_path+'/solver.prototxt',
            '_model_prototxt_path':	self.model['model_prototxt_path'],
            '_solver_log_path':	self.experiments_path+'/solver.log',
        }
        mkdirs(self.solver['snapshot_prefix'])
        self.results_path = self.experiments_path+'/results.txt'
        db_settings = getattr(self, '__' + self.db + '__')
        self.db_settings = db_settings()
        # old version: (but why?)
        #self.db_settings = {
		#	'jhmdb': self.__jhmdb__(),
		#	'vsb100': self.__vsb100__(),
		#}


    def __jhmdb__(self):
        jhmdb = {
            'db':				'jhmdb',
            'action_name':			['pour'], #['pour'],
            'level':			1,
            'video_name':			{},
            'frame':			30,
            'frame_format':			self.frame_format,
            'number_of_neighbors':		self.model['number_of_neighbors'],
            'root_path':			'/cs/vml2/mkhodaba/datasets/JHMDB/puppet_mask/{action_name}/',
            'orig_path':			'/cs/vml2/mkhodaba/datasets/JHMDB/frames/{action_name}/{video_name}/', #+frame_format
            'annotation_path':		'/cs/vml2/mkhodaba/datasets/JHMDB/puppet_mask/{action_name}/{video_name}/puppet_mask.mat',
            #'segmented_path':	'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/features/{action_name}/{video_name}/data/results/images/motionsegmentation/{level:02d}/',  #+frame_format,
            'segmented_path':		'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/segmented_frames/{action_name}/{video_name}/{level:02d}/',  #+frame_format,
            #'features_path': 	'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/features/{action_name}/{video_name}/features.txt',
            'features_path':	 	'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/features/{action_name}/{video_name}/hist.mat',
            'output_path':			'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/output/{action_name}/{video_name}/{level:02d}/{experiment_number}/', #+frame_format
            'database_path':		'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/databases/{action_name}/{video_name}/{level:02d}.h5',
            'pickle_path':			'/cs/vml2/mkhodaba/cvpr16/datasets/JHMDB/pickle/{action_name}/{video_name}/{level:02d}.p',
            'test_database_list_path':	self.experiments_path+'/database_list_{name}.txt',
            'database_list_path':		self.model['database_list_path'],
            'feature_type':			self.model['feature_type'],
        }
        start_idx = 2
        num_videos = 3 #set to None for all
        for action in jhmdb['action_name']:
        #TODO this line!
            jhmdb['video_name'][action] = getDirs(jhmdb['root_path'].format(action_name=action))[start_idx:num_videos] #TODO #TODO This should be changed!!!!!!!!!!!!!
            print '\n'.join(jhmdb['video_name'][action])
        #		for action in jhmdb['action_name']:
        #			for video in jhmdb['video_name'][action]:
        #				db_path = jhmdb['database_path'].format(action_name=action, video_name=video, level=jhmdb['level'])
        #				self.solver['_test_nets'].append(db_path)
        return jhmdb


    def __vsb100__(self):
            vsb100 = {
                'db':				'vsb100',
                'action_name':			'vw_commercial', #['pour'],
                'frame_format':			self.frame_format,
                'root_path':			'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/{action_name}/',
                'features_path': 		'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/features/{action_name}_features.mat',
                'video_info_path':		'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/features/{action_name}_vidinfo.mat',
                'output_path':			'/cs/vml2/mkhodaba/cvpr16/datasets/VSB100/output/{action_name}/', #+frame_format
                'number_of_neighbors':		self.model['number_of_neighbors'], #number of neighbors per frame
                'neighbor_frames_num':		0,
                'test_database_list_path':	self.experiments_path+'/database_list_test.txt',
                'database_list_path':		self.model['database_list_path'],
            }
            vsb100['database_path'] = '/cs/vml2/smuralid/projects/eccv16/python/databases/{action_name}' + '_' + \
                    str(self.model['number_of_neighbors']) + '_' + \
                    str(self.model['inner_product_output']) + '_' + \
                    str(vsb100['neighbor_frames_num']) +'.h5'
            self.model['number_of_neighbors'] = vsb100['number_of_neighbors'] * (2 * vsb100['neighbor_frames_num']+1) #total number of neighbors
            return vsb100


    def save(self):
        '''
        dic = {action_name:--, level:--, video_name:--}
        '''
        pickle.dump(self, open(self.experiments_path+'configs.txt', 'w'))

	#def __str__(self):
	#	from yaml import dump
	#	s = ''+\
	#	'experiment numebr: {0}\n'.format(self.experiment_number)+\
	#	'experiment path: {0}\n\n'.format(self.experiments_path)+\
	#	'model:\n{0}\n'.format(dump(self.model, default_flow_style=False, indent=4))+\
	#	'solver:\n{0}\n'.format(dump(self.solver, default_flow_style=False, indent=4))+\
	#	'db -> {0}\n'.format(self.db)
	#	db = self.db_settings[self.db]
	#	db['experiment_number'] = self.experiment_number
		#for key in db.keys():
		#	if key.endswith('_path'):
		#
		#		db[key] = db[key].format(**db)
	#	s+='db settings:\n{0}\n'.format(dump(db, default_flow_style=False, indent=4))
		#print s
	#	return s

def getConfigs(experiment_num=None):
	conf = Config(experiment_num)
	if experiment_num is not None:
		if experiment_num == -1:
			experiment_num = max([0]+map(int, getDirs(conf.experiments_root)))
		conf = pickle.load(open(conf.experiments_root+str(experiment_num)+'/configs.txt', 'r'))
	return conf


