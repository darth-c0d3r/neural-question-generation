# basic utility functions to be used everywhere

class Logger(object):

	def __init__(self, filename):
		self.terminal = sys.stdout
		self.log = open(filename, "w+")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
		self.log.flush()

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass 

def flatten_list(nested_list):
	"""
	flatten a list of lists to a single list
	"""
	return [item for sublist in nested_list for item in sublist]
