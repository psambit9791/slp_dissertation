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
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, get_scheduler

NUM_LABEL = 7
ROOT = "../"
OUTPUTS = "../outputs/blizzard_emotion/"
fname = "blizzard_metadata.csv"

device = None
if torch.cuda.is_available():
	dev = torch.cuda.current_device()
	torch.cuda.device(dev)
	device = torch.device('cuda')
	print("Using", torch.cuda.get_device_name(0))
else:
	device = torch.device("cpu")

torch.cuda.empty_cache()


def read_data(filename, sep="|"):
	data_dict = {}
	with open(filename, "r", encoding="utf-8") as f:
		for i, line_f in enumerate(f):
			if len(line_f.strip()) == 0:
				break
			meta, text, _ = line_f.split(sep)
			data_dict[meta] = text
	return data_dict

	

def load_model_checkpoint(folder, epoch, model, optimizer):
	PATH = ROOT+"model/"+folder+"/emobert-"+str(epoch)+".pt"
	checkpoint = torch.load(PATH)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['training_loss']

	return model, optimizer



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True, num_labels=NUM_LABEL)
optimizer = AdamW(model.parameters(), lr=3e-6)
model, optimizer = load_model_checkpoint("1626214249_final_emotion", 2, model, optimizer)
model.to(device)


data = read_data(fname)

id_list = []
text_list = []
pred_list = []

with torch.no_grad():
	model.eval()
	for k, v in data.items():
		tokenised = tokenizer(v, padding="max_length", truncation=True, return_tensors="pt").to(device)
		outputs = model(**tokenised)
		last_hidden_state = outputs.hidden_states[-1][:,0,:]
		np.save(OUTPUTS+str(k)+".npy", last_hidden_state.to('cpu').numpy())
		logits = outputs.logits
		pred_list += torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()
		text_list += [v]
		id_list += [k]

out_df = pd.DataFrame(data={"id": id_list, "text": text_list, "emotion": pred_list})
out_df.to_csv("../outputs/"+fname.split(".")[0]+"_emotion.csv", sep="\t", index=False)