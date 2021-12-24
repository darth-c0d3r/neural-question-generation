# contains the routine to evaluate a model

import os
import sys
import argparse
from pathlib import Path
import json5
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util import read_json, parse_unknown_args, insert_unknown_args
from util import get_device, pretty_print_results
from dataset import get_QuestionGeneration_dataloaders

def evaluate(tokenizer, model, dataloader, device, epoch, num_epochs, batch, num_batches):
	"""
	evaluate the model and return the loss value and accuracy
	also, pretty print the results
	"""

	# initiate cumulative loss and accuracy
	total_loss = 0.
	total_seqs = 0.

	# set the eval flag
	model.eval()

	with torch.no_grad():

		# iterate over all the batches
		for data_batch in tqdm(dataloader):

			source_ids = data_batch['source_ids']
			source_mask = data_batch['source_mask']
			target_ids = data_batch['target_ids'][:,:-1].contiguous()
			labels = data_batch['target_ids'][:,1:].clone().detach()
			labels[data_batch['target_ids'][:,1:] == tokenizer.pad_token_id] = -100

			source_ids, source_mask, target_ids, labels = source_ids.to(device), source_mask.to(device), target_ids.to(device), labels.to(device)

			# get the model output
			model_out = model(input_ids=source_ids, attention_mask=source_mask, decoder_input_ids=target_ids, labels=labels)

			# calculate the masked loss
			loss = model_out.loss

			# calculate the cumulative results
			total_loss += loss*len(source_ids)
			total_seqs += len(source_ids)

		total_loss /= total_seqs

	pretty_print_results("eval", epoch, num_epochs, batch, num_batches, total_loss, None, None)

	return total_loss

def main(config):
	"""
	main driver routine to set args up, load models and call the evaluation routine
	"""

	# get the device
	device = get_device(int(config["gpu_idx"]))

	# load the tokenizer and the model
	tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True)
	model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"]).to(device)

	# get the dataloaders
	dataloader = get_QuestionGeneration_dataloaders(config["dataset_file"], tokenizer, 
				config["dataset_batch_size"], config["max_src_len"], config["max_tgt_len"])

	# call the eval routine
	evaluate(tokenizer, model, dataloader, device, None, None, None, None)

if __name__ == '__main__':

	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_filename", type=str, required=True, help="config filename")
	args, unknown = parser.parse_known_args()

	# make sure that the config file exists
	assert Path(args.config_filename).is_file()

	# read the default config file
	config = read_json(args.config_filename)

	# add unknown args to config and override if needed
	unknown = parse_unknown_args(unknown)
	config = insert_unknown_args(config, unknown)

	# pretty print the config file
	print(json5.dumps(config, indent=4))
	print()

	# call the main function to set things up
	main(config)
