import numpy as np
import torch
import random
import pickle
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from datasets import load_dataset

from transformers import BertModel
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, get_scheduler


import time
import datetime
import os
import argparse
from tqdm import tqdm

NUM_LABEL = 7
dataset = "final_emotion" #Can be: goemotion, balanced_emotion

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="small dataset or full dataset")
parser.add_argument("--model", help="which model to use", default=2)
parser.add_argument("--epoch", help="load from which epoch", type=int)
parser.add_argument("--batch", help="what is the batch size", type=int, default=2)

args = parser.parse_args()

epoch = args.epoch
BATCH_SIZE = args.batch

ROOT = "../"

device = None
if torch.cuda.is_available():
    dev = torch.cuda.current_device()
    torch.cuda.device(dev)
    device = torch.device('cuda')
    print("Using", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()

def get_model_dir():
	folder = args.model + "_" + dataset
	return folder

raw_datasets = load_dataset("./"+dataset+".py")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    global tokenizer
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

test_dataset = None
if args.dataset == "small":
    test_dataset = tokenized_datasets["test"].shuffle(seed=SEED).select(range(100))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=SEED).select(range(1000))
else:
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

eval_dataset = eval_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.rename_column("label", "labels")
eval_dataset.set_format("torch")

test_dataset = test_dataset.remove_columns(["text"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch")



def flat_accuracy(preds, labels):
    pred_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_report(preds, labels):
    pred_flat = np.ravel(preds)
    labels_flat = np.ravel(labels)
    return classification_report(labels_flat, pred_flat)

def load_model_checkpoint(folder, epoch, model, optimizer):
    PATH = ROOT+"model/"+folder+"/emobert-"+str(epoch)+".pt"
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['training_loss']

    return model, optimizer


eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True, num_labels=NUM_LABEL)
optimizer = AdamW(model.parameters(), lr=3e-6)
model_folder = get_model_dir()

model, optimizer = load_model_checkpoint(model_folder, epoch, model, optimizer)
model.to(device)


predictions = []
references = []
total_test_loss = 0
with torch.no_grad():
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # last_hidden_state = outputs.hidden_states[-1][:,0,:]
        total_test_loss += outputs.loss
        logits = outputs.logits
        predictions += torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()
        references += batch["labels"].to('cpu').numpy().tolist()

avg_test_loss = total_test_loss/(len(test_dataloader)*BATCH_SIZE)
print("Validation Loss:", avg_test_loss)
print("Validation Accuracy:", flat_accuracy(predictions, references))

print("Classification Report:\n\n")
print(get_report(predictions, references))

print("\n\n")

predictions = []
references = []
total_test_loss = 0
with torch.no_grad():
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # last_hidden_state = outputs.hidden_states[-1][:,0,:]
        total_test_loss += outputs.loss
        logits = outputs.logits
        predictions += torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()
        references += batch["labels"].to('cpu').numpy().tolist()

avg_test_loss = total_test_loss/(len(test_dataloader)*BATCH_SIZE)
print("Test Loss:", avg_test_loss)
print("Test Accuracy:", flat_accuracy(predictions, references))

print("Classification Report:\n\n")
print(get_report(predictions, references))