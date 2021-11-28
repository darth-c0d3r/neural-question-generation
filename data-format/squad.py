# convert the SQuAD dataset into a consumable tsv format
# drop questions which don't have answers
# output tsv schema: context, question, answer

# === USAGE === #

# # preliminary stats on the squad dataset
# python3 squad.py --input_dir ../data/squad/raw/ --task raw_stats

# # prepare squad question answering data for consumption
# python3 squad.py --input_dir ../data/squad/raw/ --output_dir ../data/squad/processed/ --task json2tsv

# ============= #

import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

import json

def json_to_tsv(input_filename, output_filename):
	"""
	convert the squad specific json file to tsv
	drop questions that don't have answers
	"""

	# read the input json file
	with open(input_filename, 'r') as f:
		data = json.load(f)

	# open the output file
	output_file = open(output_filename, 'w+')
	output_file.write("context\tanswer\tquestion\n")

	# navigate the obscurities of the data
	print(f"\nprocessing file: {input_filename}")
	print(f"squad data version {data['version']}")
	
	data = data['data'] # list with title, paragraphs dict
	print(f"original json num rows: {len(data)}")

	# we don't need the titles
	data = [row['paragraphs'] for row in data]
	# each item is a list of questions and answers

	# keep a count of the number of rows
	count = 0

	# iterate over all the documents
	for doc in tqdm(data):

		# iterate over all the paragraphs
		for para in doc:

			# get the context of the para
			context = para['context'].replace("\n", " ").replace("\t", " ")

			# iterate over all the questions
			for ques in para['qas']:

				# ignore if ques has no answer
				if ques['is_impossible'] is True:
					continue

				# otherwise get the set of answers
				answers = list(set([ans['text'] for ans in ques['answers']]))
				answers = [ans.replace("\n", " ").replace("\t", " ") for ans in answers]

				# get the question string
				question = ques['question'].replace("\n", " ").replace("\t", " ")

				count += len(answers)
				content = [f"{context}\t{answer}\t{question}\n" for answer in answers]
				output_file.writelines(content)

	output_file.close()
	print(f"written {count} lines to {output_filename}\n")

def get_raw_stats(input_filename, output_filename=None):
	"""
	print raw stats for the squad dataset
	"""

	# read the input json file
	with open(input_filename, 'r') as f:
		data = json.load(f)

	# navigate the obscurities of the data
	print(f"\nprocessing file: {input_filename}")
	print(f"squad data version {data['version']}")
	
	data = data['data'] # list with title, paragraphs dict
	print(f"num documents: {len(data)}")

	# we don't need the titles
	data = [row['paragraphs'] for row in data]
	# each item is a list of questions and answers

	# initiate all counts
	count_para = 0
	count_ques_with_ans = 0
	count_ques_without_ans = 0
	count_ans = 0

	# iterate over all the documents
	for doc in tqdm(data):

		count_para += len(doc)

		# iterate over all the paragraphs
		for para in doc:

			# get the context of the para
			context = para['context'].replace("\n", " ").replace("\t", " ")

			# iterate over all the questions
			for ques in para['qas']:

				# ignore if ques has no answer
				if ques['is_impossible'] is True:
					count_ques_without_ans += 1
					continue

				count_ques_with_ans += 1

				# otherwise get the set of answers
				answers = list(set([ans['text'] for ans in ques['answers']]))
				answers = [ans.replace("\n", " ").replace("\t", " ") for ans in answers]

				count_ans += len(answers)

				# get the question string
				question = ques['question'].replace("\n", " ").replace("\t", " ")

	# print all the counts
	print(f"num contexts: {count_para}")
	print(f"num questions with answers: {count_ques_with_ans}")
	print(f"num questions without answers: {count_ques_without_ans}")
	print(f"num unique answers: {count_ans}")
	print()

def main(input_dir, output_dir, function):
	"""
	main driver function to set things up
	and call the extraction function on
	individual files
	"""

	input_filenames = [os.path.join(input_dir, filename) for filename 
					in os.listdir(input_dir) if filename.endswith('.json')]

	# iterate over all the files
	for filename in input_filenames:

		output_filename = None if output_dir is None else os.path.join(output_dir, Path(filename).stem + ".tsv")
		function(filename, output_filename)

if __name__ == '__main__':

	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", type=str, required=True, help="task to be performed")
	parser.add_argument("--input_dir", type=str, required=True, help="input dir with raw files")
	parser.add_argument("--output_dir", type=str, default=None, help="output dir for tsv files")
	args = parser.parse_args()

	# make sure the task is correct
	task_mapping = {
		"raw_stats": get_raw_stats,
		"json2tsv": json_to_tsv
	}

	if args.task not in task_mapping:
		print("Incorrect task.")
		sys.exit()

	# make sure that the input dir exists
	if Path(args.input_dir).is_dir() is False:
		print("Incorrect input dir.")
		sys.exit()

	# create the output dir if not exists
	if args.output_dir is not None:
		Path(args.output_dir).mkdir(exist_ok=True, parents=True)

	# call the main folder to set things up
	main(args.input_dir, args.output_dir, task_mapping[args.task])
