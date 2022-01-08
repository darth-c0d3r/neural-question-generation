# use this script to benchmark tokenizers and models 
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
# vary batch sizes, sequence lengths
# compare fast and slow tokenizer
# ======================================================== #

# ======================================================== #
# HOW TO BENCHMARK A MODEL

# Fix an input of n words
# Fix a tokenizer
# get times for generation
# vary batch sizes, sequence lengths
# compare across generation algos
# like beam size, length_penalty etc.s
# ======================================================== #


# probably don't change these unless you have good reason
RANDOM_SEED = 42
WORDLIST_FILENAME = "wordlist.txt"

import os
import argparse
from time import time
import itertools
import shutil

import random
random.seed(RANDOM_SEED)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

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

def benchmark_model(model, batch, decode_params):
	"""
	just do a forward pass of the model
	with a decoding algorithm
	no need to do detokenization as it
	is already benchmarked in tokenizer
	"""

	NUM_ITERS = 10

	start = time()

	for _ in range(NUM_ITERS):

		model_out = model.generate(**batch, **decode_params)

	end = time()

	return (end-start)/NUM_ITERS;

def main(args):
	"""
	the main driver function to run everything
	"""

	do_tokenizer = False
	do_model = False
	do_model_quantized = False
	do_model_distilled = True

	model_name = "sshleifer/distilbart-cnn-6-6"
	device = torch.device(args.gpu_idx if args.gpu_idx >= 0 else 'cpu')
	print(f"using device {device}")
	print(f"using model {model_name}")

	# ========================= #
	# benchmark tokenizers here #
	# ========================= #

	if do_tokenizer is True:
		print("\nBenchmarking Tokenizer.\n")

		batch_size = [8,16,32,64]
		sequence_length = [128, 256, 512, 1024]

		tokenizer_norm = AutoTokenizer.from_pretrained(model_name, use_fast=False)
		tokenizer_fast = AutoTokenizer.from_pretrained(model_name, use_fast=True )

		for bs, sl in itertools.product(*[batch_size, sequence_length]):

			input_text = get_input_text(bs, sl)

			print(f"Norm Tokenizer | BS {bs} | SL {sl} | Time {benchmark_tokenizer(tokenizer_norm, input_text, sl):.4f}")
			print(f"Fast Tokenizer | BS {bs} | SL {sl} | Time {benchmark_tokenizer(tokenizer_fast, input_text, sl):.4f}")


	# ================================= #
	# benchmark transformer models here #
	# ================================= #

	if do_model is True:
		print("\nBenchmarking Model.\n")

		batch_size = [16,32]
		sequence_length = [256,512]
		beam_size = [1,4]

		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
		model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

		# Add ONNX / Quantization / Whatever support here

		for bs, sl in itertools.product(*[batch_size, sequence_length]):

			input_text = get_input_text(bs, sl)
			source = tokenizer(input_text, max_length=sl, padding='longest', truncation=True, return_tensors='pt')

			batch = {'input_ids': source['input_ids'].to(device), 
						'attention_mask': source['attention_mask'].to(device)}

			decode_params = { # NON DEFAULT PARAMS HERE
				"min_length": 8, "length_penalty": 1.0,
			}

			for beam in beam_size:

				decode_params["num_beams"] = beam
				time = benchmark_model(model, batch, decode_params)

				print(f"Model | BS {bs} | SL {sl} | Beams {beam} | Time {time:.4f}")


	# ================================= #
	# benchmark transformer quantized models here #
	# ================================= #

	if do_model_quantized is True:
		print("\nBenchmarking quantized Model.\n")

		batch_size = [16,32]
		sequence_length = [256,512]
		beam_size = [1,4]

		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
		model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
		model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

		for bs, sl in itertools.product(*[batch_size, sequence_length]):

			input_text = get_input_text(bs, sl)
			source = tokenizer(input_text, max_length=sl, padding='longest', truncation=True, return_tensors='pt')

			batch = {'input_ids': source['input_ids'].to(device), 
						'attention_mask': source['attention_mask'].to(device)}

			decode_params = { # NON DEFAULT PARAMS HERE
				"min_length": 8, "length_penalty": 1.0,
			}

			for beam in beam_size:

				decode_params["num_beams"] = beam
				time = benchmark_model(model, batch, decode_params)

				print(f"Quant Model | BS {bs} | SL {sl} | Beams {beam} | Time {time:.4f}")

	# ================================= #
	# benchmark transformer distilled models here #
	# ================================= #

	if do_model_distilled is True:
		print("\nBenchmarking distilled Models.\n")

		encoders = [1,3,6]
		decoders = [1,3,6]
		batch_size = 16
		sequence_length = 256
		beam_size = 4

		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

		input_text = get_input_text(batch_size, sequence_length)
		source = tokenizer(input_text, max_length=sequence_length, padding='longest', truncation=True, return_tensors='pt')

		batch = {'input_ids': source['input_ids'].to(device), 
					'attention_mask': source['attention_mask'].to(device)}

		decode_params = { # NON DEFAULT PARAMS HERE
			"min_length": 8, "length_penalty": 1.0, "num_beams": beam_size
		}

		for e, d in itertools.product(*[encoders, decoders]):

			config = AutoConfig.from_pretrained(model_name)
			config.encoder_layers = e
			config.decoder_layers = d
			model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to(device)

			time = benchmark_model(model, batch, decode_params)
			model.save_pretrained("temp")
			size = os.stat("temp/pytorch_model.bin").st_size
			shutil.rmtree("temp")

			print(f"Distilled Model | Enc {e} | Dec {d} | Time {time:.4f} | Size {size/1000000} MB")


if __name__ == '__main__':

	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu_idx", type=int, default=-1, help="device id to be used")
	args = parser.parse_args()

	main(args)
