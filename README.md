# Introduction

[HuggingFace](https://huggingface.co/) is one of the most useful libraries for a NLP researcher / developer as it provides numerous pre-trained models, datasets, and tons of utility functions for NLP. In this repository, I'm trying to setup a complete pipeline for a Machine Learning project and the task I've chosen for the setup is Question Generation for Paragraphs. This is a seq2seq task for which I intend to fine-tune a pre-trained encoder-decoder Transformer model for Extractive Summarization like BART / Pegasus. More specifically, I'm finetuning the `sshleifer/distilbart-cnn-6-6` model on the SQuAD dataset.

# Features / Goals

* Environment setup using YAML file
* Hyper-parameter management with configs [done]
* Efficient data loading using LMDB [done]
* Dataset Visualization / Stats [done]
* Results Visualization / Stats [done]
* LR Scheduler [done]
* Multiple Decoding Algorithm Options [done]
* Intermediate Checkpoints [done]
* Parallel Logging to file [done]
* Use Fast Tokenizers [done]
* Latency + Efficiency Benchmarking [done]
* Distributed Training and Inference
* ONNX Optimization [not implemented in hgfc]
* Model Quantization [done]
* Model Distillation
* Hosting using Streamlit / Gradio
* Deploying on HuggingFace Hub [done]

# Dataset

The goal of Question Generation is to generate a valid and fluent question according to a given passage and the target answer. Hence, the input to the model will be a passage context and an answer, and the output / target will be the question for the given answer. Question Generation can be used in many scenarios, such as automatic tutoring systems, improving the performance of Question Answering models and enabling chat-bots to lead a conversation. The final dataset is created by taking the union of the following Question Answering Datasets. The dataset must have the following three columns: context, question, answer.

## [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. We use the SQuAD 1.1 variant which does not have unanswerable questions. So, every question will have a corresponding answer and vice-versa.

### Preprocessing

The first step is to remove questions which don't have answers. After that, we split the train set into Train and Eval sets and treat the dev set as the test set.

### Stats

**Original Dataset**

| Split | Num Docs | Num Contexts | Ques w/ Ans | Ques w/o Ans | Num Unique Ans |
| ----- | -------- | ------------ | ----------- | ------------ | -------------- |
| Train | 442      | 19035        | 86821       | 43498        | 86821          |
| Dev   | 35       | 1204         | 5928        | 5945         | 10279          |

**After Preprocessing**

| Split | Num Rows |   Context  | Answer | Question |
| ----- | -------- | ---------- | ------ | -------- |
| Train | 80995    | 653,120,20 | 43,3,1 | 40,10,1  | 
| Eval  | 5826     | 445,123,67 | 28,3,1 | 29,10,3  |
| Test  | 10297    | 629,129,25 | 29,4,1 | 31,10,3  |

The numbers in the columns indicate max, avg, min number of words.

## [Natural Questions](https://ai.google.com/research/NaturalQuestions) [Not Used]

The Natural Questions corpus is a question answering dataset by Google. Each example is comprised of a google.com query and a corresponding Wikipedia page. Each Wikipedia page has a passage (or long answer) annotated on the page that answers the question and one or more short spans from the annotated passage containing the actual answer. The long and the short answer annotations can however be empty. If they are both empty, then there is no answer on the page at all. If the long answer annotation is non-empty, but the short answer annotation is empty, then the annotated passage answers the question but no explicit short answer could be found. Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”, instead of a list of short spans.

## [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) [Not Used]

TriviaQA is a realistic text-based question answering dataset which includes 950K question-answer pairs from 662K documents collected from Wikipedia and the web. This dataset is more challenging than standard QA benchmark datasets such as Stanford Question Answering Dataset (SQuAD), as the answers for a question may not be directly obtained by span prediction and the context is very long. TriviaQA dataset consists of both human-verified and machine-generated QA subsets.

# Directory Structure

```bash
|-- README.md
|-- data
|	|-- squad
|	|	|-- raw
|	|	|	|-- train-v2.0.json
|	|	|	|-- dev-v2.0.json
|	|	|-- processed
|	|	|	|-- [processed data files]
|	|	|	|-- splits
|	|	|	|	|-- train.tsv
|	|	|	|	|-- eval.tsv
|	|	|	|	|-- test.tsv
|	|	|	|	|-- lmdb*
|	|	|	|	|	|-- train.lmdb*
|	|	|	|	|	|-- eval.lmdb*
|	|	|	|	|	|-- test.lmdb*
|-- data-format
|	|-- squad.py
|-- data-utils
|	|-- proto
|	|	|-- data_item.proto
|	|	|-- data_item_pb2.py*
|	|-- data_stats.py
|	|-- create_lmdb.py
|-- src
|	|-- util.py
|	|-- plotting.py
|	|-- dataset.py
|	|-- distribute.py
|	|-- train.py
|	|-- evaluate.py
|	|-- predict.py
|	|-- generate.py
|	|-- distill.py
|-- config
|	|-- train.config
|	|-- eval.config
|	|-- pred.config
|-- vis
|	|-- tsv_viewer.py
|-- stats
|	|-- [stats related files]
|-- benchmark
|	|-- benchmark.py
|	|-- wordlist.txt
|	|-- results.txt
|-- logs
|	|-- train
|	|	|-- run_4 [20 epochs, no scheduler]
|	|	|-- run_6 [4 epochs and scheduler]
|	|-- pred
|	|	|-- run_2 [default decoding params]
|	|	|-- run_3 [adjusted decoding params]
|	|	|-- run_4 [dynamic quantized pred]

* : file created programmatically
```

# Commands to run

```bash

# preliminary stats on the squad dataset
python3 squad.py --input_dir ../data/squad/raw/ --task raw_stats

# prepare squad question answering data for consumption by converting to tsv
python3 squad.py --input_dir ../data/squad/raw/ --output_dir ../data/squad/processed/ --task json2tsv

# at this point manually split the train set into train and eval in ./splits

# fetch initial stats on the dataset
python3 data_stats.py --input_path ../data/squad/processed/splits/ --output_path ../stats/squad/

# take a look at a few samples of the dataset
streamlit run tsv_viewer.py -- --input_path ../data/squad/processed/splits/eval.tsv

# convert the tsv data into lmdb database for efficient loading
python3 -m grpc_tools.protoc -I./proto --python_out=./proto ./proto/data_item.proto
python3 create_lmdb.py --input_path ../data/squad/processed/splits/ --output_path ../data/squad/processed/splits/lmdb/

# training routine [adjust params in config]
# set --gpus 0 for CPU training
python3 distribute.py --filename train.py --config_filename ../config/train.config --nodes 1 --gpus 4 --rank 0
python3 train.py --config_filename ../config/train.config

# evaluation routine [adjust params in config]
python3 evaluate.py --config_filename ../config/eval.config

# get predictions [adjust params in config]
python3 predict.py --config_filename ../config/pred.config

# to get interactive predictions [adjust params in config]
python3 generate.py --config_filename ../config/pred.config

# view the results using streamlit
streamlit run tsv_viewer.py -- --input_path ../logs/pred/run_6/eval.tsv

```

# Huggingface Hub

1. Instructions on uploading model to hub. [Link](https://huggingface.co/docs/transformers/model_sharing)
	- Remember to update the default config file with required decoding parameters.
2. Link to my finetuned 6-6 QGen Model. [Link](https://huggingface.co/gpssohi/distilbart-qgen-6-6)
