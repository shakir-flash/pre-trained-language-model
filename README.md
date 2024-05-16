# Fine-tuning Pre-trained Language Models for ComVE Shared Task

## Introduction
This repository contains code and instructions for fine-tuning pre-trained language models on SubTasks A, B, and C of the ComVE shared task from SemEval-2020. Each subtask involves different aspects of common sense reasoning and requires fine-tuning a pre-trained model for specific tasks.

# Setup your environment and prepare data

First, carefully follow the *General Instructions for Programming Assignments*.

To install the libraries required for this assignment run:

    pip install -r requirements.txt

or in case you work with macOS:

    pip install -r requirements-macos.txt

Clone the MeasEval repository:

    git clone https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation.git SemEval2020-Task4-Data


## lmA: Identifying Nonsensical Statements
### Overview
SubTask A evaluates whether a system can distinguish if a natural language statement makes sense to humans or not. Given two similar statements, the task is to select the nonsensical one. This task involves binary classification based on common sense understanding.

### Setup and Execution
To fine-tune a pre-trained model for SubTask A:
1. **Setup**: Install necessary dependencies and clone the repository.
2. **Data Preparation**: Load and preprocess the dataset using pandas and the Datasets library.
3. **Model Loading**: Load the pre-trained RoBERTa model and its tokenizer.
4. **Fine-tuning**: Use the Trainer API from Transformers to fine-tune the model on the dataset.
5. **Prediction and Evaluation**: After fine-tuning, make predictions on the test dataset and evaluate using accuracy metrics.

### Repository Details
- RoBERTa model is fine-tuned using the Hugging Face Transformers library.
- Data pre-processing involves loading and tokenizing the dataset using pandas and the Datasets library.
- Trainer API from Transformers is used for fine-tuning, with evaluation metrics computed on the test set.

## lmB: Selecting Reasons Against Common Sense
### Overview
SubTask B involves selecting the reason that explains why a given statement is against common sense from three possible reasons. This task requires understanding common sense principles and selecting the most appropriate reason.

### Setup and Execution
To fine-tune a pre-trained model for SubTask B:
1. **Setup**: Install dependencies, clone the repository, and prepare the data.
2. **Model Loading**: Load the pre-trained RoBERTa model and its tokenizer.
3. **Data Loading**: Load the dataset for training, development, and testing, translating the labels into numerical indices.
4. **Data Preprocessing**: Tokenize the statement-reason pairs, handling padding, truncation, and unflattening of the tokenized sequences.
5. **Fine-tuning Process**: Train the model on the dataset using the Trainer object, with evaluation on the development set after each epoch.

### Repository Details
- RoBERTa model, a variant of BERT, is fine-tuned using the Transformers library.
- Data pre-processing includes loading and tokenizing the dataset, handling padding, truncation, and unflattening of tokenized sequences.

## lmC: Generating Reasons for Nonsensical Statements
### Overview
SubTask C focuses on generating reasons for nonsensical statements provided in the dataset using a pre-trained sequence-to-sequence language model. This task requires the model to understand the context of the statement and generate a plausible reason for its nonsensical nature.

### Setup and Execution
To fine-tune a pre-trained model for SubTask C:
1. **Setup**: Install dependencies, clone the repository, and prepare the data.
2. **Model Loading**: Load the pre-trained BART model and its tokenizer.
3. **Data Loading**: Load the dataset for training, development, and testing, organizing it into appropriate DataFrame structures.
4. **Data Preprocessing**: Tokenize the nonsensical statements and reasons, preparing them for input to the model.
5. **Fine-tuning Process**: Train the model on the dataset using the Trainer object, with evaluation on the development set after each epoch. Evaluate the model's performance using BLEU and ROUGE metrics on the test set.

### Repository Details
- BART, a pre-trained sequence-to-sequence model, is fine-tuned using the Hugging Face Transformers library.
- Data pre-processing involves tokenizing nonsensical statements and reasons, preparing them for input to the model.
- Evaluation includes computing BLEU and ROUGE metrics on the test set.

## Conclusion
This repository provides a comprehensive pipeline for fine-tuning pre-trained language models on the ComVE shared task, covering SubTasks A, B, and C. For detailed information and instructions, please refer to the respective sections in this README.
