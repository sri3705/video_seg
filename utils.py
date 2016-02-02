from os import makedirs, listdir
from os.path import isdir, dirname

def mkdirs(path):
	dir_path = dirname(path)
	try:
		makedirs(dir_path)
	except Exception as e:	
		pass

def getDirs(path):
	try:
		return sorted([o for o in listdir(path) if isdir(path+'/'+o) and o[0] != '.'])
	except Exception as e:
		print e
		print 'Error in getDirs'
		return []

def getFuncArgNames(func):
	return func.func_code.co_varnames[:func.func_code.co_argcount]
