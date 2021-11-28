# after converting the dataset to a fixed format
# use this script to fetch stats about it
# can add more tests to it as needed

# Usage : python3 data_stats.py --input_path ../data/squad/processed/splits/ --output_path ../stats/squad/

import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import csv
import numpy as np

import matplotlib.pyplot as plt

sys.path.append("../src/")
from util import flatten_list

def plot_bargraph(labels_counts, output_dir, data_name):
	"""
	plot and save a bar graph for labels -> counts mapping
	"""
	output_filename = os.path.join(output_dir, data_name + "_labels.png")
	plt.bar(list(labels_counts.keys()), list(labels_counts.values()), color='grey')
	plt.xlabel('label id')
	plt.ylabel('count')
	plt.title(f'label counts in {data_name}')
	plt.savefig(output_filename)
	plt.clf()
	# plt.show()

	return

def plot_histogram(lengths, output_dir, data_name):
	"""
	plot and save a bar graph for labels -> counts mapping
	"""

	output_filename = os.path.join(output_dir, data_name + "_lengths.png")
	plt.hist(lengths, rwidth=0.9, color='green')
	plt.xlabel('input len')
	plt.ylabel('count')
	plt.title(f'input length distribution in {data_name}')
	plt.savefig(output_filename)
	plt.clf()
	# plt.show()

	return

def get_stats_file(output_dir, filename):
	"""
	analyze and print stats for filename
	columns in file: context, question, answer
	to plot: distribution of num. words in each

	"""

	print(f"\nanalyzing {filename} ...")

	# specify the list of columns to be plotted here
	cols_to_use = ["context", "answer", "question"]

	# init the lengths here
	col_length = {col: [] for col in cols_to_use}

	chunksize = 1024 # to read the tsv file in chunks as it may be too large
	data = pd.read_csv(filename, sep="\t", header=0, chunksize=chunksize, quoting=csv.QUOTE_NONE)

	for chunk in tqdm(data):

		for col in cols_to_use:

			# get the column data
			col_data = chunk[col].tolist()

			# get the length / num. words
			col_data = [len(str(row).strip().split()) for row in col_data]

			# append to main list
			col_length[col].extend(col_data)

	# iterate over all the columns to print stats
	for col in cols_to_use:

		print(f"stats for col {col}")
		print(f"num rows: {len(col_length[col])}")
		print(f"avg. words: {sum(col_length[col])/len(col_length[col]):.2f}")
		print(f"max. words: {max(col_length[col])}")
		print(f"min. words: {min(col_length[col])}")

		plot_histogram(col_length[col], output_dir, Path(filename).stem + f"_{col}")
		print(f"plot saved to {output_dir}")
		print()

	print("="*10)

def main(input_dir, output_dir):
	"""
	main function to run and get the stats
	"""

	filenames = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".tsv")]
	for filename in filenames:
		get_stats_file(output_dir, filename)

if __name__ == '__main__':

	# setup argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_path", type=str, required=True, help="input path for the processed data directory.")
	parser.add_argument("--output_path", type=str, required=True, help="output directory for stats plots.")
	args = parser.parse_args()

	# check existense of input path
	if Path(args.input_path).is_dir() is False:
		print("Input path is invalid.")
		sys.exit()

	# create the output directory safely
	Path(args.output_path).mkdir(parents=True, exist_ok=True)

	# call the main function
	main(args.input_path, args.output_path)
