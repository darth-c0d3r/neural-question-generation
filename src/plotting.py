# use this for all the plotting related routines

import os
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Plotter(object):
	"""
	main Plotter class that handles all plotting routines
	"""
	
	def __init__(self, logs_folder, filename, title, lines, samples_per_epoch):
		"""
		logs_folder is the base folder for logging
		filename is the name of the file for curr plot
		title is the title of the plot
		lines is a list of strings for names of plot lines
		"""

		# initialize the base class explicitly
		super().__init__()

		self.filename = os.path.join(logs_folder, f"plots/{filename}")
		Path(self.filename).parent.mkdir(parents=False, exist_ok=True)

		self.title = title
		self.data = {linename: [] for linename in lines}
		self.samples_per_epoch = samples_per_epoch

	def extend_plot(self, new_values):
		"""
		add new values to the data and update the plot
		new_values is a str -> list dict with same keys as self.data
		"""

		for linename in new_values:
			self.data[linename].extend(new_values[linename])

		self.make_plot()

	def make_plot(self):
		"""
		use self.data to create the plots
		"""

		# plot all the lines
		for linename in self.data:

			plt.plot([(i+1)/self.samples_per_epoch for i in range(len(self.data[linename]))], self.data[linename],
					label=linename)

		# plot dotted v-lines
		num_epochs = max([len(self.data[linename]) for linename in self.data]) // self.samples_per_epoch
		for xc in range(1, num_epochs+1):
			plt.axvline(x=xc, linestyle='--', color='grey')

		# some post processing
		plt.xlabel('epochs')
		plt.ylabel(Path(self.filename).stem)
		plt.title(self.title)
		plt.legend()

		# save the figure and clear the buffer
		plt.savefig(self.filename)
		plt.clf()

def run_tests():

	import random
	from time import sleep

	plotter = Plotter(".", "temp.png", "random values", ["curr", "running"], 10)

	avg = 0

	for cnt in range(100):
		sleep(1)
		curr = random.randint(0,20)
		avg = (avg*cnt + curr)/(cnt+1)
		cnt += 1
		plotter.extend_plot({"curr": [curr], "running": [avg]})
		print(cnt, curr, avg)
