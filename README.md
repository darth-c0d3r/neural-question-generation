# Introduction

[HuggingFace](https://huggingface.co/) is one of the most useful libraries for a NLP researcher / developer as it provides numerous pre-trained models, datasets, and tons of utility functions for NLP. In this repository, I'm trying to setup a complete pipeline for a Machine Learning project and the task I've chosen for the setup is Question Generation for Paragraphs. This is a seq2seq task for which I intend to fine-tune pre-trained encoder-decoder Transformer models for Extractive Summarization like BART / Pegasus.

# Features / Goals

* Environment setup using YAML file
* Hyper-parameter management with configs [done]
* Efficient data loading using LMDB [done]
* Dataset Visualization / Stats [done]
* Results Visualization / Stats [done]
* LR Scheduler
* Multiple Decoding Algorithm Options
* Intermediate Checkpoints [done]
* Parallel Logging to file [done]
* Latency + Efficiency Benchmarking
* Distributed Training and Inference
* Model Distillation
* Model Quantization
* ONNX Optimization
* Hosting using Streamlit / Gradio
* Deploying on HuggingFace Hub

# Dataset

The goal of Question Generation is to generate a valid and fluent question according to a given passage and the target answer. Hence, the input to the model will be a passage context and an answer, and the output / target will be the question for the given answer. Question Generation can be used in many scenarios, such as automatic tutoring systems, improving the performance of Question Answering models and enabling chat-bots to lead a conversation. The final dataset is created by taking the union of the following Question Answering Datasets. The dataset must have the following three columns: context, question, answer.

## [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. We use the SQuAD 1.1 variant which does not have unanswerable questions. So, every question will have a corresponding answer and vice-versa.

### Preprocessing

The first step is to remove questions which don't have answers. After that, we split the train set into Train and Eval sets and treat the dev set as the test set.

### Stats

<b>Original Dataset</b>

<u>Train Set</u>
num documents: 442
num contexts: 19035
num questions with answers: 86821
num questions without answers: 43498
num unique answers: 86821

<u>Dev Set</u>
num documents: 35
num contexts: 1204
num questions with answers: 5928
num questions without answers: 5945
num unique answers: 10279

<b>After Preprocessing</b>

analyzing ../data/squad/processed/splits/eval.tsv ...
6it [00:00, 33.32it/s]
stats for col context
num rows: 5826
avg. words: 123.12
max. words: 445
min. words: 67
plot saved to ../stats/squad/

stats for col answer
num rows: 5826
avg. words: 2.97
max. words: 28
min. words: 1
plot saved to ../stats/squad/

stats for col question
num rows: 5826
avg. words: 10.04
max. words: 29
min. words: 3
plot saved to ../stats/squad/

==========

analyzing ../data/squad/processed/splits/test.tsv ...
11it [00:00, 37.60it/s]
stats for col context
num rows: 10279
avg. words: 129.28
max. words: 629
min. words: 25
plot saved to ../stats/squad/

stats for col answer
num rows: 10279
avg. words: 3.62
max. words: 29
min. words: 1
plot saved to ../stats/squad/

stats for col question
num rows: 10279
avg. words: 10.33
max. words: 31
min. words: 3
plot saved to ../stats/squad/

==========

analyzing ../data/squad/processed/splits/train.tsv ...
80it [00:02, 30.73it/s]
stats for col context
num rows: 80995
avg. words: 119.55
max. words: 653
min. words: 20
plot saved to ../stats/squad/

stats for col answer
num rows: 80995
avg. words: 3.18
max. words: 43
min. words: 1
plot saved to ../stats/squad/

stats for col question
num rows: 80995
avg. words: 10.07
max. words: 40
min. words: 1
plot saved to ../stats/squad/

==========


## [Natural Questions](https://ai.google.com/research/NaturalQuestions) [Currently not in use.]

The Natural Questions corpus is a question answering dataset by Google. Each example is comprised of a google.com query and a corresponding Wikipedia page. Each Wikipedia page has a passage (or long answer) annotated on the page that answers the question and one or more short spans from the annotated passage containing the actual answer. The long and the short answer annotations can however be empty. If they are both empty, then there is no answer on the page at all. If the long answer annotation is non-empty, but the short answer annotation is empty, then the annotated passage answers the question but no explicit short answer could be found. Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”, instead of a list of short spans.

## [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) [Currently not in use.]

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
|	|-- dataset.py
|-- vis
|	|-- tsv_viewer.py
|-- stats
|	|-- [stats related files]

* : file created programmatically
```

# Commands to run

```bash

# preliminary stats on the squad dataset
python3 squad.py --input_dir ../data/squad/raw/ --task raw_stats

# prepare squad question answering data for consumption
python3 squad.py --input_dir ../data/squad/raw/ --output_dir ../data/squad/processed/ --task json2tsv

# at this point manually split the train set into train and eval in ./splits

# fetch initial stats on the dataset
python3 data_stats.py --input_path ../data/squad/processed/splits/ --output_path ../stats/squad/

# take a look at a few samples of the dataset
streamlit run tsv_viewer.py -- --input_path ../data/squad/processed/splits/eval.tsv

# convert the tsv data into lmdb database for efficient loading
python3 -m grpc_tools.protoc -I./proto --python_out=./proto ./proto/data_item.proto
python3 create_lmdb.py --input_path ../data/squad/processed/splits/ --output_path ../data/squad/processed/splits/lmdb/

# training routing
python3 train.py --config_filename ../config/train.config

# evaluation routing
python3 evaluate.py --config_filename ../config/eval.config

```

# ToDo / Points to Ponder

* Have a Start token for decoder input?
