# use this script to benchmark tokenizers, models, decoding algos
# feel free to add own tests here
# pending support for onnx, quantization, fp16 etc

# every time aritrary n words are needed
# read words from wordlist.txt
# use a fixed seed and sample n random words
# for consistency and reproducability

# ======================================================== #
# HOW TO BENCHMARK A TOKENIZER

# Fix an input of n words
# get times for encoder and decoder
# vary sequence lengths
# compare fast and slow tokenizer
# ======================================================== #

# probably don't change these unless you have good reason
RANDOM_SEED = 42
WORDLIST_FILENAME = "wordlist.txt"

import argparse
from time import time

import random
random.seed(RANDOM_SEED)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_input_text(batch_size, sequence_length):
	"""
	return a list of batch_size number of sequences
	each sequence is just a random permutation of base
	"""

	# make sure you ain't killing the system
	assert sequence_length <= 4096

	with open(WORDLIST_FILENAME, 'r') as f:
		words = f.readlines()

	words = [w.strip() for w in words[:sequence_length]]
	sequences = [words.copy() for _ in range(batch_size)]
	for idx in range(batch_size):
		random.shuffle(sequences[idx])
		sequences[idx] = ' '.join(sequences[idx])

	return sequences

def benchmark_tokenizer(tokenizer, input_text, max_length):
	"""
	the simple goal here is to just take the input text
	and tokenize - detokenize it a bunch of times
	and return the average time taken
	"""

	NUM_ITERS = 10

	start = time()

	for _ in range(NUM_ITERS):

		source = tokenizer(input_text, max_length=max_length, padding='longest', truncation=True, return_tensors='pt')
		output = tokenizer.batch_decode(source['input_ids'], skip_special_tokens=True)

	end = time()

	return (end - start)/NUM_ITERS

def main(args):
	"""
	the main driver function to run everything
	"""

	do_tokenizer = True


	model_name = "sshleifer/distilbart-cnn-6-6"
	device = torch.device(args.gpu_idx if args.gpu_idx >= 0 else 'cpu')
	print(f"using device {device}")
	print(f"using model {model_name}")

	# ========================= #
	# benchmark tokenizers here #
	# ========================= #

	if do_tokenizer is True:
		print("Benchmarking Tokenizer.\n")

		batch_size = [8,16,32,64]
		sequence_length = [128, 256, 512, 1024]

		tokenizer_norm = AutoTokenizer.from_pretrained(model_name, use_fast=False)
		tokenizer_fast = AutoTokenizer.from_pretrained(model_name, use_fast=True )

		for bs in batch_size:
			for sl in sequence_length:

				input_text = get_input_text(bs, sl)

				print(f"Norm Tokenizer | BS {bs} | SL {sl} | Time {benchmark_tokenizer(tokenizer_norm, input_text, sl):.4f}")
				print(f"Fast Tokenizer | BS {bs} | SL {sl} | Time {benchmark_tokenizer(tokenizer_fast, input_text, sl):.4f}")

if __name__ == '__main__':

	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu_idx", type=int, default=-1, help="device id to be used")
	args = parser.parse_args()

	main(args)
