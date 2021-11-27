# Introduction

[HuggingFace](https://huggingface.co/) is one of the most important libraries for a NLP researcher / developer as it provides numerous pretrained models, datasets, and tons of utility functions for NLP. In this repository, I'm trying to setup a complete pipeline for a Machine Learning project and the task I've chosen for the setup is Question Generation for Paragraphs. This is a seq2seq task for which I intend to finetune pretrained encoder-decoder Transformer models for Extractive Summarization like BART.

# Features / Goals

* Environment setup using YAML file
* Hyperparameter management with configs
* Efficient data loading using LMDB
* Dataset Visualization / Stats
* Results Visualization / Stats
* Intermediate Checkpoints
* Parallel Logging to file
* Timing Benchmarking
* Distributed Training and Inference
* Model Distillation
* Model Quantization
* ONNX Optimization
* Hosting using Streamlit / Gradio
* Deploying on Huggingface Hub
