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
    df = pd.read_csv(ROOT+"data/imdb/csv/"+mode+".csv")
    return df

# Get the dialog from the generated dataframe
def get_cell(df, row, column_name):
    return df.loc[df.index[row], column_name]


def save_to_sent(data, mode, dataset="emo"):
    with open(ROOT+"data/imdb/sent/"+mode+"_"+dataset+".pkl", "wb") as f:
        pickle.dump(data, f)



dataset = load_dataset('imdb')
# print(dataset)
temp = dataset["test"].train_test_split(test_size=0.5)
dataset["test"] = temp["test"]
dataset["validation"] = temp["train"]
del dataset["unsupervised"]

def split_data_into_lists(mode):
    global dataset
    temp_data = dataset[mode]
    rows = []
    
    for i, d in enumerate(temp_data):      
        rows.append([d["text"], d["label"]])
            
    df = pd.DataFrame(data=rows, columns=['text', 'label'])
    df.to_csv(ROOT+"data/imdb/csv/"+mode+".csv", index=False)

### UNCOMMENT IF THE DATA SPLITTING PROCESS IS CHANGED
split_data_into_lists("test")
split_data_into_lists("validation")
split_data_into_lists("train")


train = load_csv("train")
val = load_csv("validation")
test = load_csv("test")


def get_unprocessed_row(df, row):
    
    sentence_c = get_cell(df, row, "text")
    label = get_cell(df, row, "label")
    
    return sentence_c, label



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