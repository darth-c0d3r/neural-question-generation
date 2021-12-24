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
from dataset import get_QuestionGeneration_dataloaders

def predict(tokenizer, model, dataloader, device, config):
	"""
	get predictions from the model and output them to output_filename
	"""

	temp_filename = os.path.join(config["logs_folder"], "temp.tsv")
	open(temp_filename, 'w+').close()
	header = True

	# set the eval flag
	model.eval()

	with torch.no_grad():

		# iterate over all the batches
		for data_batch in tqdm(dataloader):

			source_ids = data_batch['source_ids'].to(device)
			source_mask = data_batch['source_mask'].to(device)
			batch = {'input_ids': source_ids, 'attention_mask': source_mask}
			
			# get the model output
			model_out = model.generate(**batch, **config["decoding_params"]) # add output hyperparams here
			model_out = tokenizer.batch_decode(model_out, skip_special_tokens=True)

			# put the outputs in a dataframe and write to file
			df = pd.DataFrame(model_out, columns=["Prediction"]).fillna("[EMPTY]")
			df.to_csv(temp_filename, sep="\t", header=header, index=False, mode="a")
			header = False

	merge_input_output_files(config["dataset_file"], temp_filename, config["output_filename"])
	print(f"Written predictions to {config['output_filename']}")

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

	# call the pred routine
	predict(tokenizer, model, dataloader, device, config)

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

	# get unique path for base folder and create it
	config["logs_folder"] = get_unique_path(config["base_folder"], "pred", prefix="run")
	Path(config["logs_folder"]).mkdir(parents=False, exist_ok=False)

	# get the output filename
	config["output_filename"] = os.path.join(config["logs_folder"], f"{Path(config['dataset_file']).stem}.tsv")
	
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
