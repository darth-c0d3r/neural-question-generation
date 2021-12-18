# contains classes for datasets and dataloaders
# create 2 types: returns tokens and returns embeddings
# first make 1 and then promote it to 2

import os
import sys
from pathlib import Path

import pandas as pd
import csv
import lmdb

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append("../data-utils/proto/")
from data_item_pb2 import DataItem

class QuestionGeneration_TSV_Dataset(Dataset):
	"""
	class to encode and return text and labels from tsv dataset
	"""

	def __init__(self, tsv_filename, tokenizer, max_src_len, max_tgt_len, data_size=None):
		"""
		tsv_filename is the path to the tsv file
		use data_size if you need to limit the dataset size
		"""

		# initialize the base class explicitly
		super().__init__()

		self.tsv_filename = tsv_filename
		self.tokenizer = tokenizer
		self.max_src_len = max_src_len
		self.max_tgt_len = max_tgt_len

		self.data = pd.read_csv(self.tsv_filename, sep="\t", header=0, quoting=csv.QUOTE_NONE)
		self.length = len(self.data)
		if data_size is not None:
			self.length = min(data_size, self.length)

		print(f"tsv dataset {Path(self.tsv_filename).stem} size {self.length}.")

	def __len__(self):
		return self.length

	def __getitem__(self, index):

		row = self.data.iloc[index]
		context = row.context
		answer = row.answer
		question = row.question

		return {'context': context, 'answer': answer, 'question': question}

	def collate_fn(self, data):

		source = [f"{d['answer']} <s> {d['context']}" for d in data]
		target = [d["question"] for d in data]

		source = self.tokenizer(source, max_length=self.max_src_len, padding='longest', truncation=True, return_tensors='pt')
		target = self.tokenizer(target, max_length=self.max_tgt_len, padding='longest', truncation=True, return_tensors='pt')

		source_ids = source['input_ids']
		source_mask = source['attention_mask']
		target_ids = target['input_ids']
		
		return {
			'source_ids': source_ids.to(dtype=torch.long),
			'source_mask': source_mask.to(dtype=torch.long),
			'target_ids': target_ids.to(dtype=torch.long),
		}

class QuestionGeneration_LMDB_Dataset(QuestionGeneration_TSV_Dataset):
	"""
	class to encode and return text and labels from lmdb dataset
	"""

	def __init__(self, lmdb_filename, tokenizer, max_src_len, max_tgt_len, data_size=None):
		"""
		lmdb_filename is the path to the lmdb database
		use data_size if you need to limit the dataset size
		"""

		# initialize the base class explicitly
		# super().__init__()

		self.lmdb_filename = lmdb_filename
		self.tokenizer = tokenizer
		self.max_src_len = max_src_len
		self.max_tgt_len = max_tgt_len

		env = lmdb.open(self.lmdb_filename, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
		with env.begin(write=False) as txn:
			self.length = txn.stat()['entries']
			if data_size is not None:
				self.length = min(data_size, self.length)
			print(f"lmdb dataset {Path(self.lmdb_filename).stem} size {self.length}.")

		self.txn = env.begin(write=False)
	
	def __len__(self):
		return self.length

	def __getitem__(self, index):

		serialized_str = self.txn.get(f'{index}'.encode('ascii'))
		data_item = DataItem()
		data_item.ParseFromString(serialized_str)

		context = data_item.context
		answer = data_item.answer
		question = data_item.question

		return {'context': context, 'answer': answer, 'question': question}

def get_QuestionGeneration_dataloaders(base_path, tokenizer, batch_size, 
				max_src_len=512, max_tgt_len=128, use_tsv=False, num_workers=2):
	"""
	build and return dataloaders for QuestionGeneration datasets
	# base_path is the directory containing tsv / lmdb files
	# this follows a strict format as in README.md
	# batch size is batch size duh
	"""

	# lonely tsv file
	if base_path.endswith(".tsv"):
		data_files = [base_path]
		use_tsv = True

	# lonely lmdb file
	elif base_path.endswith(".lmdb"):
		data_files = [base_path]
		use_tsv = False

	# full fledged folder
	else:

		# lmdb inside folder
		if "lmdb" in os.listdir(base_path) and use_tsv is False:
			data_files = [os.path.join(base_path, f"lmdb/{filename}") for filename 
						in os.listdir(os.path.join(base_path, "lmdb")) if filename.endswith(".lmdb")]
			use_tsv = False

		# tsv inside folder
		else:
			data_files = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if filename.endswith(".tsv")]
			use_tsv = True

	if use_tsv is True: data = { Path(filename).stem : QuestionGeneration_TSV_Dataset(filename, tokenizer, max_src_len, max_tgt_len) for filename in data_files }
	else: data = { Path(filename).stem : QuestionGeneration_LMDB_Dataset(filename, tokenizer, max_src_len, max_tgt_len) for filename in data_files }

	dataloaders = {name: DataLoader(data[name], batch_size=batch_size, shuffle=(name=="train"),
					num_workers=num_workers, pin_memory=True, collate_fn=data[name].collate_fn) for name in data}


	if len(dataloaders) == 1: return list(dataloaders.values())[0]
	return dataloaders

def run_tests():

	from transformers import AutoTokenizer
	model_name = "google/pegasus-xsum"
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	base_path = "../data/squad/processed/splits/"
	dataloaders_tsv = get_QuestionGeneration_dataloaders(base_path, tokenizer, 4, use_tsv=True)
	dataloaders_lmdb = get_QuestionGeneration_dataloaders(base_path, tokenizer, 4, use_tsv=False)

	for data in dataloaders_tsv['test']:
		
		source_ids = data['source_ids']
		source_mask = data['source_mask']
		target_ids = data['target_ids'][:,:-1].contiguous()
		labels = data['target_ids'][:,1:].clone().detach()
		labels[data['target_ids'][:,1:] == tokenizer.pad_token_id] = -100

		print(source_ids.shape, source_mask.shape, target_ids.shape, labels.shape)
		print(target_ids)
		print(labels)

		break

	print()

	for data in dataloaders_lmdb['test']:
		
		source_ids = data['source_ids']
		source_mask = data['source_mask']
		target_ids = data['target_ids'][:,:-1].contiguous()
		labels = data['target_ids'][:,1:].clone().detach()
		labels[data['target_ids'][:,1:] == tokenizer.pad_token_id] = -100

		print(source_ids.shape, source_mask.shape, target_ids.shape, labels.shape)

		break

# run_tests()