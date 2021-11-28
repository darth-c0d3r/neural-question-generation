"""
Used to create LMDB dataset from .tsv files for faster data loading.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import lmdb
from proto import data_item_pb2 # change this according to your dataset

import pandas as pd
import csv

def get_tsv_reader(input_filename, chunksize):
    """
    read the input tsv files into a dataframe
    """

    return pd.read_csv(input_filename, sep='\t', header=0, chunksize=chunksize, quoting=csv.QUOTE_NONE)

def tsv_to_lmdb(input_filename, output_filename, container='protobuf'):
    """
    read the tsv file and write it to a lmdb database
    """

    # set up the output file and env
    LMDB_MAP_SIZE = 1 << 40
    env = lmdb.open(output_filename, map_size=LMDB_MAP_SIZE)

    idx = 0
    # start the transaction
    with env.begin(write=True) as txn:

        # iterate over all the chunks of the input tsv file
        for chunk in tqdm(get_tsv_reader(input_filename, chunksize=1024)):

            # chunk = chunk.fillna("EMPTY")
            for _, row in chunk.iterrows():

                # enter the reqd columns of each row
                data_item = data_item_pb2.DataItem()
                data_item.context = row['context']
                data_item.answer = str(row['answer'])
                data_item.question = row['question']

                key = f'{idx}'.encode('ascii')
                txn.put(
                    key,
                    data_item.SerializeToString()
                    )
                idx += 1

    return idx

def main(input_path, output_path):
    """
    main driver function to convert the dataset to lmdb
    """

    # get the list of dataset files to be analyzed
    if Path(input_path).is_dir():
        dataset_filenames = [os.path.join(input_path, filename) for filename in os.listdir(input_path) if filename.endswith(".tsv")]
    else:
        dataset_filenames = [input_path]

    # call the function for all the files
    for filename in dataset_filenames:
        print(f"Processing {filename}")
        output_filename = os.path.join(output_path, Path(filename).stem + ".lmdb")
        num_rows = tsv_to_lmdb(filename, output_filename, container='protobuf')
        print(f"Written {num_rows} rows to {output_filename}")

if __name__ == "__main__":

    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="input path to a tsv dataset file / directory")
    parser.add_argument("--output_path", type=str, required=True, help="output directory for lmdb dataset")
    args = parser.parse_args()

    # check existense of input path
    if (Path(args.input_path).is_dir() or Path(args.input_path).is_file())  is False:
        print("Input path is invalid.")
        sys.exit()

    # create the output directory safely
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # call the main function with the arguments
    main(args.input_path, args.output_path)