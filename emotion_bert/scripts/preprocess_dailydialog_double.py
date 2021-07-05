import numpy as np
import torch
import random
import pickle
import pandas as pd

from datasets import load_dataset


ROOT = "../"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if torch.cuda.is_available():
    dev = torch.cuda.current_device()
    torch.cuda.device(dev)
    print("Using", torch.cuda.get_device_name(0))
else:
    torch.device("cpu")


def load_csv(mode):
    df = pd.read_csv(ROOT+"data/daily_dialog_double/csv/"+mode+".csv")
    return df

# Get the dialog from the generated dataframe
def get_cell(df, row, column_name):
    return df.loc[df.index[row], column_name]



### Save tokenised data

def save_to_pkl(data, mode, dataset="emo"):
    with open(ROOT+"data/daily_dialog_double/pkl/"+mode+"_"+dataset+".pkl", "wb") as f:
        pickle.dump(data, f)

def save_to_sent(data, mode, dataset="emo"):
    with open(ROOT+"data/daily_dialog_double/sent/"+mode+"_"+dataset+".pkl", "wb") as f:
        pickle.dump(data, f)



dataset = load_dataset('daily_dialog')
# print(dataset)

def split_data_into_lists(mode):
    global dataset
    temp_data = dataset[mode]
    rows = []
    max_length_previous = 0
    max_pidx = -1
    max_length_current = 0
    max_cidx = -1
    
    for i, d in enumerate(temp_data):
        acts = d["act"]
        dialogs = d["dialog"]
        emotions = d["emotion"]
        
        for idx in range(1, len(acts)):
            rows.append([dialogs[idx-1], dialogs[idx], acts[idx], emotions[idx]])
            
    df = pd.DataFrame(data=rows, columns=['previous_dialog', 'current_dialog', 'act', 'emotion'])
    df.to_csv(ROOT+"data/daily_dialog_double/csv/"+mode+".csv", index=False)

### UNCOMMENT IF THE DATA SPLITTING PROCESS IS CHANGED
split_data_into_lists("test")
split_data_into_lists("validation")
split_data_into_lists("train")


train = load_csv("train")
val = load_csv("validation")
test = load_csv("test")


def get_unprocessed_row(df, row, mode="emo"):
    global  max_previous_length
    global max_current_length
    
    sentence_p = get_cell(df, row, "previous_dialog")
    sentence_c = get_cell(df, row, "current_dialog")
    
    label = None
    
    if mode == "act":
        label = get_cell(df, row, "act")
    else:
        label = get_cell(df, row, "emotion")
    
    return sentence_p, sentence_c, label

# Emo data

train_data = []
for i in range(train.shape[0]):
    train_data.append(get_unprocessed_row(train, i))
save_to_sent(train_data, "train")

val_data = []
for i in range(val.shape[0]):
    val_data.append(get_unprocessed_row(val, i))
save_to_sent(val_data, "val")

test_data = []
for i in range(test.shape[0]):
    test_data.append(get_unprocessed_row(test, i))
save_to_sent(test_data, "test")


# Act data

train_data = []
for i in range(train.shape[0]):
    train_data.append(get_unprocessed_row(train, i, "act"))
save_to_sent(train_data, "train", "act")

val_data = []
for i in range(val.shape[0]):
    val_data.append(get_unprocessed_row(val, i, "act"))
save_to_sent(val_data, "val", "act")

test_data = []
for i in range(test.shape[0]):
    test_data.append(get_unprocessed_row(test, i, "act"))
save_to_sent(test_data, "test", "act")