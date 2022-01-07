# contains the routine to train / fine-tune a hgfc model

import argparse
import json5
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util import read_json, parse_unknown_args, insert_unknown_args, get_unique_path
from util import get_device, pretty_print_results, save_model, Logger, setup_multiprocessing

from dataset import get_QuestionGeneration_dataloaders
from evaluate import evaluate
from plotting import Plotter

def train(tokenizer, model, dataloaders, optimizer, scheduler, device, config):
	"""
	the main training routine
	train the model
	on data from dataloaders
	using optimizer and scheduler
	config for other params
	"""

	print("\n" + "="*10 + "TRAINING LOGS" + "="*10 + "\n")

	# initialize the plotters
	loss_plotter = Plotter(config["logs_folder"], "loss.png", "Loss Values", ["train", "eval"], config["num_evals_per_epoch"])
	
	# get the eval frequency
	num_batches = int(np.ceil(len(dataloaders["train"].dataset)/config["dataset_batch_size"]))
	eval_every_batch = (num_batches + 1) // config["num_evals_per_epoch"]

	# initiate cumulative loss
	total_loss = 0.
	total_seqs = 0.
	best_eval_loss = 1e5 # init large best loss INF

	# iterate over all the epochs
	for epoch in tqdm(range(1, config["num_epochs"]+1)):

		# iterate for all minibatches
		for batch_idx, data_batch in enumerate(dataloaders["train"]):

			# set the train flag
			model.train()

			source_ids = data_batch['source_ids']
			source_mask = data_batch['source_mask']
			target_ids = data_batch['target_ids'][:,:-1].contiguous()
			labels = data_batch['target_ids'][:,1:].clone().detach()
			labels[data_batch['target_ids'][:,1:] == tokenizer.pad_token_id] = -100

			source_ids, source_mask, target_ids, labels = source_ids.to(device), source_mask.to(device), target_ids.to(device), labels.to(device)

			# flush the gradients
			optimizer.zero_grad()

			# get the model output
			model_out = model(input_ids=source_ids, attention_mask=source_mask, decoder_input_ids=target_ids, labels=labels)

			# calculate the masked loss
			loss = model_out.loss

			# update the parameters
			loss.backward()
			optimizer.step()
			if scheduler is not None: scheduler.step()

			# calculate the cumulative results
			total_loss += loss*len(source_ids)
			total_seqs += len(source_ids)

			# it's eval time!
			if batch_idx % eval_every_batch == 0:

				pretty_print_results("train", epoch, config["num_epochs"], batch_idx+1, 
							num_batches, loss, total_loss, total_seqs)

				eval_loss = evaluate(tokenizer, model, dataloaders["eval"], device, epoch, config["num_epochs"], batch_idx+1, num_batches)

				# save intermediate checkpoints
				save_model(model, config["ckpts_folder"], "latest")

				# extend plots
				loss_plotter.extend_plot({"train": [loss.detach().cpu()], "eval": [eval_loss.cpu()]})
				
				# save best model separately
				if eval_loss < best_eval_loss:
					best_eval_loss = eval_loss

					# copyfile(os.path.join(config["ckpts_folder"], "model_latest.pt"), os.path.join(config["ckpts_folder"], "model_best.pt"))
					save_model(model, config["ckpts_folder"], "best")
					print("best model so far")

				print()

		# basic eval after the epoch
		_ = evaluate(tokenizer, model, dataloaders["eval"], device, epoch, config["num_epochs"], "EPOCH", "END")

		# save intermediate checkpoints
		save_model(model, config["ckpts_folder"], f"epoch")
		print()

def main(config):
	"""
	main driver routine to set args up, load models, datasets, and call the training routine
	"""

	# get the device

	if config["gpu_idx"] != -1: # GPU training
		gpu_rank = setup_multiprocessing(config["gpu_idx"])
		device = get_device(config["gpu_idx"])
		torch.cuda.set_device(device)
	else:
		device = get_device(config["gpu_idx"])

	return

	# load the tokenizer and the model
	tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True)
	model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"]).to(device)

	# get the dataloaders
	dataloaders = get_QuestionGeneration_dataloaders(config["dataset_dir"], tokenizer, 
					config["dataset_batch_size"], config["max_src_len"], config["max_tgt_len"])

	# declare the optimizers and LR schedulers
	optimizer = optim.Adam(list(model.parameters()), lr=config["learning_rate"], betas=(0.9, 0.999))

	num_batches = int(np.ceil(len(dataloaders["train"].dataset)/config["dataset_batch_size"]))
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_batches, gamma=0.5)
	# scheduler = None

	# call the train routine
	# train(tokenizer, model, dataloaders, optimizer, scheduler, device, config)


if __name__ == '__main__':

	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_filename", type=str, required=True, help="config filename")
	
	# --local_rank is needed for torch 1.2.0 compatibility only
	parser.add_argument("--local_rank", type=int, default=-1, help="gpu_idx placeholder")
	args, unknown = parser.parse_known_args()

	# make sure that the config file exists
	assert Path(args.config_filename).is_file()

	# read the default config file
	config = read_json(args.config_filename)

	# add unknown args to config and override if needed
	unknown = parse_unknown_args(unknown)
	config = insert_unknown_args(config, unknown)

	# get unique path for base folder and create it
	config["logs_folder"] = get_unique_path(config["base_folder"], "train", prefix="run")
	Path(config["logs_folder"]).mkdir(parents=False, exist_ok=False)

	# set up the checkpoints folder
	config["ckpts_folder"] = os.path.join(config["logs_folder"], "ckpts")
	Path(config["ckpts_folder"]).mkdir(parents=False, exist_ok=False)

	# set gpu_idx = local_rank

	# # this is for newer versions of torch
	# if 'LOCAL_RANK' not in os.environ: os.environ['LOCAL_RANK'] = '-1'
	# config["gpu_idx"] = int(os.environ['LOCAL_RANK'])

	config["gpu_idx"] = int(args.local_rank)

	# save the current config file in the logs folder
	with open(os.path.join(config["logs_folder"], Path(args.config_filename).name), "w+") as f:
		json5.dump(config, f, indent=4)

	# set up parallel logging
	sys.stdout = Logger(os.path.join(config["logs_folder"], "logfile.log"))

	# pretty print the config file
	print(json5.dumps(config, indent=4))
	print()

	# call the main function to set things up
	main(config)
