import numpy as np
import torch
import random
import pickle
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from datasets import load_dataset

from transformers import BertModel
from transformers import get_linear_schedule_with_warmup


import time
import datetime
import os


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
	folder = str(int(datetime.datetime.now().timestamp()))
	os.mkdir(ROOT+"model/"+folder)
	return folder


def load_from_pkl(mode, dataset):
    data = None
    with open(ROOT+"data/daily_dialog/pkl/"+mode+"_"+dataset+".pkl", "rb") as f:
        data = pickle.load(f)
    return data

def load_from_sent(mode, dataset):
    data = None
    with open(ROOT+"data/daily_dialog/sent/"+mode+"_"+dataset+".pkl", "rb") as f:
        data = pickle.load(f)
    return data

def get_sentence_pairs(dataset, tokenizer):
    global device
    inputs = []
    attention_masks = []
    labels = []

    #######################################
    max_steps = min(len(dataset), 100)  ####
    #######################################
    
    for i in dataset[:max_steps]:
        enc_s = tokenizer.encode_plus(text=i[0], 
                                      text_pair=i[1], 
                                      add_special_tokens = True, 
                                      padding = 'max_length',
                                      truncation="only_first", 
                                      pad_to_max_length = True, 
                                      return_attention_mask = True, 
                                      return_tensors = 'pt')
        inputs.append(enc_s["input_ids"])
        attention_masks.append(enc_s["attention_mask"])
        labels.append(i[2])
    
    inputs = torch.cat(inputs, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
        
    return TensorDataset(inputs, attention_masks, labels)

train_data = load_from_sent("train", "emo")
val_data = load_from_sent("val", "emo")
test_data = load_from_sent("test", "emo")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_dataset = get_sentence_pairs(train_data, tokenizer)
val_dataset = get_sentence_pairs(val_data, tokenizer)


BATCH_SIZE = 2

train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler = RandomSampler(train_dataset), # Select batches randomly
        batch_size = BATCH_SIZE # Trains with this batch size.
    )

validation_dataloader = DataLoader(
        val_dataset, # The validation samples.
        sampler = RandomSampler(val_dataset), # Pull out batches randomly.
        batch_size = BATCH_SIZE # Evaluates with this batch size.
    )



class EmotionBERTModel(nn.Module):
    def __init__(self):
        super(EmotionBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear1 = nn.Linear(768, 256)
        self.final = nn.Linear(256, 7)
    
    def forward(self, ids, mask):
        sequence_output, pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)
        output = F.relu(self.linear1(sequence_output[:,0,:].view(-1,768))) ## extract the 1st token's embeddings
        output = self.final(output)
        return output


model = EmotionBERTModel()
model.to(device)

### Setting up hyperparameters
epochs = 5

criterion = nn.CrossEntropyLoss() ## If required define your own criterion
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
# Create the learning rate scheduler.
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def save_model_checkpoint(folder, model, optimizer, epoch, loss):
	PATH = ROOT+"model/"+folder+"/emobert-"+str(epoch)+".pt"
	torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)


model_folder = create_model_dir()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    
    model.train()
    
    t0 = time.time()
    total_train_loss = 0
    
    for step, batch in enumerate(train_dataloader):

        # Progress update every 1000 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = time.time() - t0
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        
        outputs = model(b_input_ids, b_input_mask)
        
        loss = criterion(outputs, b_labels)
        total_train_loss += loss
        
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader) 
    training_time = time.time() - t0
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            outputs = model(b_input_ids, b_input_mask,)
        
            loss = criterion(outputs, b_labels)
        
        # Accumulate the validation loss.
        total_eval_loss += loss
        
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(outputs.to('cpu').numpy(), b_labels.to('cpu').numpy())
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = time.time() - t0
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    save_model_checkpoint(model_folder, model, optimizer, epoch_i, avg_train_loss)