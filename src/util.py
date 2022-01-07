# basic utility functions to be used everywhere

import subprocess
import json5
import os
import sys
from pathlib import Path

import torch

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

def get_num_lines(filename):
	"""
	return the number of lines in filename
	"""

	output = subprocess.run(["wc", "-l", filename], capture_output=True).stdout.split()[0]
	return int(output)

def get_device(gpu_idx):
	"""
	return the device to be used
	"""

	device = torch.device('cpu') if gpu_idx < 0 or not torch.cuda.is_available() else torch.device(gpu_idx)
	print(f"Using Device {device}")
	return device

def setup_multiprocessing(gpu_idx):
	"""
	call this to setup distributed training
	"""

	torch.distributed.init_process_group(
		backend = 'nccl',
		# init_method = 'env://',
		# world_size = '?',
		# rank = '?',
	)

	print(f"World Size: {dist.get_world_size()} \t Local Rank: {dist.get_rank()} \t GPU Idx {gpu_idx}")

	return dist.get_rank()

def read_json(json_filename):
	"""
	read a json file and return a dict
	supports json with comments
	"""

	with open(json_filename, "r") as f:
		data = json5.load(f)
	return data

def parse_unknown_args(args):
	"""
	parse unknown args returned by argparse
	"""

	args = " ".join(args).split("--")
	args = [x.strip() for x in args if len(x.strip()) > 0]
	args = [x.split() for x in args]

	assert all([len(x) > 1 for x in args])

	args = {x[0]: x[1:] if len(x) > 2 else x[1] for x in args}

	return args

def insert_unknown_args(config, args):
	"""
	insert all the values from args to config
	override if needed
	try to retain original dtype
	"""

	for arg_key in args:

		try:
			config[arg_key] = type(config[arg_key])(args[arg_key])
			print(f"overriding {arg_key} in config")
		except:
			config[arg_key] = args[arg_key]

	return config

def get_unique_path(base_folder, task_name, prefix="run"):
	"""
	return a unique path for subfolder task_name in base_folder
	"""

	base_folder = os.path.join(base_folder, task_name)

	# create the base dir if it doesn't exist
	Path(base_folder).mkdir(parents=True, exist_ok=True)

	# get the list of relevent folders
	dir_num = max([int(dir_name.lstrip(f"{prefix}_")) for dir_name in os.listdir(base_folder) if dir_name.startswith(prefix)]+[0]) + 1

	# return the new base directory
	return os.path.join(base_folder, f"{prefix}_{dir_num}")

def save_model(model, ckpt_dir, model_name):
	"""
	save the huggingface model checkpoint
	"""

	output_dir = os.path.join(ckpt_dir, model_name)
	model.save_pretrained(output_dir)
	print(f"saved the model to {output_dir}")

def pretty_print_results(tag, epoch, num_epochs, batch, num_batches,
						curr_loss, total_loss, total_seqs):
	"""
	pretty print the output results
	"""

	epoch_component = tag
	if epoch is not None:
		epoch_component += f" epoch {epoch}/{num_epochs}"

	batch_component = None
	if batch is not None:
		batch_component = f"batch {batch}/{num_batches}"

	curr_component = f"loss: {curr_loss: .3f}"

	running_component = None
	if total_loss is not None:
		running_component = f"running loss: {total_loss/total_seqs: .3f}"

	all_components = [epoch_component, batch_component, curr_component, running_component]
	all_components = "\t".join([c for c in all_components if c is not None])

	print(all_components)

def merge_input_output_files(src_filename_1, src_filename_2, output_filename):
	"""
	merge input and temp files
	temp file can be tsv or lmdb
	implement lmdb later maybe
	"""

	src_1 = open(src_filename_1, "r")
	src_2 = open(src_filename_2, "r")
	tgt = open(output_filename, "w+")

	while True:
		l1 = src_1.readline()
		l2 = src_2.readline()

		if len(l1) == 0 or len(l2) == 0:
			break

		o = l1.rstrip("\n") + "\t" + l2.rstrip("\n") + "\n"
		tgt.write(o)

	src_1.close()
	src_2.close()
	tgt.close()
