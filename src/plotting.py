# use this for all the plotting related routines

import os
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Plotter(object):
	"""
	main Plotter class that handles all plotting routines
	"""
	
	def __init__(self, logs_folder, filename, title, lines, samples_per_epoch, opt_fn):
		"""
		logs_folder is the base folder for logging
		filename is the name of the file for curr plot
		title is the title of the plot
		lines is a list of strings for names of plot lines
		opt_fn is the function to track the "best" value
		eg. for loss it will be min, for accuracy max
		"""

		# initialize the base class explicitly
		super().__init__()

		self.filename = os.path.join(logs_folder, f"plots/{filename}")
		Path(self.filename).parent.mkdir(parents=False, exist_ok=True)

		self.title = title
		self.data = {linename: [] for linename in lines}
		self.samples_per_epoch = samples_per_epoch
		self.opt_fn = opt_fn

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

			X = [(i+1)/self.samples_per_epoch for i in range(len(self.data[linename]))]
			Y = self.data[linename]

			if len(Y) > 0: y_opt, x_opt = self.opt_fn(zip(Y, X))
			else: y_opt, x_opt = 0., 0.

			plt.plot(X, Y, label=f"{linename} [{y_opt:.3f}]")
			plt.plot([x_opt], [y_opt], 'k.')

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
