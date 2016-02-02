from enum import Enum
class LogType(Enum):
	PRINT = 1
	FILE = 2
class Logger:
	def __init__(self, log_type=LogType.PRINT, log_path=''):
		self.type = log_type
		if log_type == LogType.FILE:
			self.file = open(log_path,'w')

	def log(self,message):
		if self.type == LogType.PRINT:
			print message
		else:
			self.file.write(message+'\n')
			print message
			#TODO file
	def close(self):
		if hasattr(self, "file"):
			self.file.close()
		#TODO close file 
