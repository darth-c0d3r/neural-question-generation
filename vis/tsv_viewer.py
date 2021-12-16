# a generic script to view a tsv file
# assume that the tsv file can be large
# add any features that you like!

# planned features:
# show / hide columns
# sort rows by column value
# ^ expensive operation
# sort by index / shuffle
# show n results

import argparse
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import csv
import numpy as np
import random

sys.path.append("../src/")
from util import get_num_lines

st.set_page_config(layout="wide")
ROWS_PER_PAGE = 16

@st.cache
def load_df(input_filename, idx_chunk):
	"""
	load the dataframe in a cached manner
	"""

	df = pd.read_csv(input_filename, sep='\t', header=0, 
			quoting=csv.QUOTE_NONE, chunksize=ROWS_PER_PAGE)

	curr = 0
	for chunk in df:
		if curr == idx_chunk:
			return chunk
		curr += 1

def main(input_filename):
	"""
	main function to display tsv file
	"""

	st.write(f"<h3> Displaying {input_filename} </h3>", unsafe_allow_html=True)
	num_rows = get_num_lines(input_filename)-1 # remove 1 for the header
	num_chunks = int(np.ceil(num_rows/ROWS_PER_PAGE))

	display_type = st.selectbox('Display Type:', ['shuffled', 'indexed'])

	if display_type == 'indexed':
		inp = st.text_input(f"Chunk Index. [0, {num_chunks-1}]")
		idx_chunk = int(inp) if len(inp.strip()) != 0 else 0
	else:
		idx_chunk = random.randint(0, num_chunks-1)

	if idx_chunk >= num_chunks: idx_chunk = 0
	df = load_df(input_filename, idx_chunk)

	st.write(f"Displaying Chunk {idx_chunk}/{num_chunks-1}.", unsafe_allow_html=True)
	st.write(df.to_html(), unsafe_allow_html=True)

if __name__ == "__main__":

	# setup argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_path", type=str, required=True, help="input path to a tsv dataset file")
	args = parser.parse_args()

	# check existense of input path
	if Path(args.input_path).is_file()  is False:
		print("Input path is invalid.")
		sys.exit()
	
	main(args.input_path)	
