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

from distutils import util


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="small dataset or full dataset")
parser.add_argument("--epoch", help="how many eopchs should this run for", type=int, default=5)
parser.add_argument("--batch", help="what is the batch size", type=int, default=2)
parser.add_argument("--save", help="save the models per epoch", type=util.strtobool, default=False)

args = parser.parse_args()

num_epochs = args.epoch
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

def create_model_dir():
    folder = str(int(datetime.datetime.now().timestamp())) + "_imdb"
    os.mkdir(ROOT+"model/"+folder)
    return folder


raw_datasets = load_dataset("imdb")
temp = raw_datasets["test"].train_test_split(test_size=0.5)
raw_datasets["test"] = temp["test"]
raw_datasets["validation"] = temp["train"]
del raw_datasets["unsupervised"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    global tokenizer
    return tokenizer(examples["text"], padding="max_length", truncation=True)

        

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_dataset = None
eval_dataset = None
# test_dataset = None
if args.dataset == "small":
    train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(100))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=SEED).select(range(100))
else:
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

train_dataset = train_dataset.remove_columns(["text"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")

eval_dataset = eval_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.rename_column("label", "labels")
eval_dataset.set_format("torch")



def flat_accuracy(preds, labels):
    pred_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def save_model_checkpoint(folder, model, optimizer, epoch, tr_loss, val_loss):
    PATH = ROOT+"model/"+folder+"/emobert-"+str(epoch)+".pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': tr_loss,
            'validation_loss': val_loss
            }, PATH)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-6)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

if args.save:
    model_folder = create_model_dir()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    predictions = []
    references = []
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += outputs.loss
        loss.backward()

        logits = outputs.logits
        predictions += torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()
        references += batch["labels"].to('cpu').numpy().tolist()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    avg_train_loss = total_train_loss/(len(train_dataloader)*BATCH_SIZE)
    print("Training Loss:", avg_train_loss)
    print("Training Accuracy:", flat_accuracy(predictions, references))

    # torch.cuda.empty_cache()

    predictions = []
    references = []
    total_val_loss = 0
    with torch.no_grad():
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_val_loss += outputs.loss
            logits = outputs.logits
            predictions += torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()
            references += batch["labels"].to('cpu').numpy().tolist()

    avg_val_loss = total_val_loss/(len(eval_dataloader)*BATCH_SIZE)
    print("Validation Loss:", avg_val_loss)
    print("Validation Accuracy:", flat_accuracy(predictions, references))

    if args.save:
        save_model_checkpoint(model_folder, model, optimizer, epoch, avg_train_loss, avg_val_loss)
    