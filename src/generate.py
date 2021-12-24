# contains the routine to get predictions from a model

import os
import sys
import argparse
from pathlib import Path
import json5
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util import read_json, parse_unknown_args, insert_unknown_args, get_unique_path
from util import get_device, pretty_print_results, merge_input_output_files, Logger

def predict(tokenizer, model, device, config):
	"""
	get predictions from the model and output them to output_filename
	"""

	# set the eval flag
	model.eval()

	with torch.no_grad():

		while True:

			print(f"\n{'='*20}\n")

			source = input("Input Passage: ")
			source = tokenizer([source], max_length=config["max_src_len"], padding='longest', truncation=True, return_tensors='pt')

			batch = {'input_ids': source['input_ids'].to(device), 
						'attention_mask': source['attention_mask'].to(device)}
			
			# get the model output
			model_out = model.generate(**batch, **config["decoding_params"]) # add output hyperparams here
			model_out = tokenizer.batch_decode(model_out, skip_special_tokens=True)

			print(f"\nOutput: {model_out}")

def main(config):
	"""
	main driver routine to set args up, load models and call the evaluation routine
	"""

	# get the device
	device = get_device(int(config["gpu_idx"]))

	# load the tokenizer and the model
	tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True)
	model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"]).to(device)

	# call the pred routine
	predict(tokenizer, model, device, config)

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
