print "experiment setups initiating ... "
from configs import *
from NetworkFactory import *
from DatabaseGenerator import *
from Logger import *
import sys



def setup_experiment(extract_features=False, visualization=False):
	# need to extract features?
	config = getConfigs()
	print "Experiment number:", config.experiment_number
	logger = Logger(config.log_type, config.log_path)
	logger.log('Configs created:')
	logger.log(str(config))
	logger.log('Creating Network ...')
	createNetwork(config.model)
	logger.log('Adding test layers ...')
	addTestLayers(config)
	logger.log('Creating Solver prototxt ...')
	with open(config.solver['_solver_prototxt_path'], 'w') as f:
		for key, val in config.solver.iteritems():
			if not key.startswith('_'):
				f.write('{key}:{val}\n'.format(key=key, val='"{0}"'.format(val) if isinstance(val, str) else val))

	if extract_features:
		#TODO: ^^^^ add neighbor_num to db_settigs shit!
		print 'extract features'
		createDatabase(config.db, config.db_settings, logger)
		#TODO create the database list
		#TODO: probably in configs need to set how to merge them: for now separately
	else:
		write_db_list(config.db_settings, logger)
	logger.close()

	#TODO save configs
	config.save()

if __name__=='__main__':
	if "-f" in sys.argv:
		extract_features = True
	else:
		extract_features = False
	print "extract_features = ", extract_features
	setup_experiment(extract_features=extract_features, visualization=False)
