import numpy as np
import torch
import random
import pickle
import pprint
import pandas as pd

from datasets import load_dataset
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

ROOT = "../"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

split_keys = ["train", "validation", "test"]
label_list = [0, 1, 2, 3, 4, 5, 6]


min_words_dd = {"train": 1000, "validation": 120, "test": 300}
min_words_md = {"train": 1000, "validation": 150, "test": 300}

dd = {}
md = {}
dd_counts = None
md_counts = None 

dyda_e = load_dataset('silicone', 'dyda_e')
meld_e = load_dataset('silicone', 'meld_e')


def plot_data_distribution(data_dict, label_var, filename=None):
    global split_keys
    counts = {}
    fig, ax = plt.subplots(1, 3, figsize=(8,8))

    for i, k in enumerate(split_keys):
        counts[k] = data_dict[k][label_var].value_counts().to_dict()
        ax[i].bar(counts[k].keys(), counts[k].values())
        ax[i].set_ylabel("Number of occurences")
        ax[i].set_xlabel("Emotion")
        ax[i].set_title("Data for " + str(k))

    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(ROOT+"images/"+filename+".png")
        plt.clf()
    else:
        plt.show()

    return counts


def clean_dyda():
    global dd, dd_counts, min_words_dd, split_keys
    for k in split_keys:
        emotions_with_large_data = []
        for emo in label_list:
            if dd_counts[k][emo] > min_words_dd[k]:
                emotions_with_large_data.append(emo)
        to_delete = []
        for idx, row in dd[k].iterrows():
            if row["Label"] in emotions_with_large_data and row['Num_Tokens'] <= 4:
                to_delete.append(idx)
        dd[k].drop(to_delete, inplace=True)
        dd[k].drop(columns=['Dialogue_ID', 'Emotion', 'Idx', 'Num_Tokens'], inplace=True)
        dd[k].rename(columns={'Utterance': 'text', 'Label': 'label'}, inplace=True)
    return dd

def clean_meld():
    global md, md_counts, min_words_md, split_keys
    for k in split_keys:
        emotions_with_large_data = []
        for emo in label_list:
            if md_counts[k][emo] > min_words_md[k]:
                emotions_with_large_data.append(emo)
        to_delete = []
        for idx, row in md[k].iterrows():
            if row["Label"] in emotions_with_large_data and row['Num_Tokens'] <= 4:
                to_delete.append(idx)
        md[k].drop(to_delete, inplace=True)
        md[k].drop(columns=['Speaker', 'Dialogue_ID', 'Utterance_ID', 'Emotion', 'Idx', 'Num_Tokens'], inplace=True)
        md[k].rename(columns={'Utterance': 'text', 'Label': 'label'}, inplace=True)
    return md


def add_token_length(df):
    lengths = []
    for idx, row in df.iterrows():
        words= [word for word in word_tokenize(str(row['Utterance'])) if word.isalnum()]
        lengths.append(len(words))
    df['Num_Tokens'] = lengths
    return df

for i in dyda_e.keys():
    dd[i] = pd.DataFrame(dyda_e[i])
    dd[i] = add_token_length(dd[i])
    md[i] = pd.DataFrame(meld_e[i])
    md[i] = add_token_length(md[i])

dd_counts = plot_data_distribution(dd, 'Label', 'dyda_init')
md_counts = plot_data_distribution(md, 'Label', 'meld_init')

clean_dyda()
clean_meld()

dd_counts = plot_data_distribution(dd, 'label', 'dyda_cleaned')
md_counts = plot_data_distribution(md, 'label', 'meld_cleaned')


def get_ratios(counts):

    tr_val = {}
    tr_ts = {}

    for lbl in label_list:
        tr_val[lbl] = counts["train"][lbl]/counts["validation"][lbl]
        tr_ts[lbl] = counts["train"][lbl]/counts["test"][lbl]
    
    return tr_val, tr_ts

dd_val_ratio, dd_test_ratio = get_ratios(dd_counts)
print("~~~~ DYDA Ratios ~~~~")
print("    Training:Validation :", np.median(list(dd_val_ratio.values())))
# print(pd.DataFrame(list(zip(list(dd_val_ratio.keys()), list(dd_val_ratio.values()))), columns=['Label', 'Ratio']))
print("    Training:Test :", np.median(list(dd_test_ratio.values())))
# print(pd.DataFrame(list(zip(list(dd_test_ratio.keys()), list(dd_test_ratio.values()))), columns=['Label', 'Ratio']))

print()

md_val_ratio, md_test_ratio = get_ratios(md_counts)
print("~~~~ MELD Ratios ~~~~")
print("    Training:Validation :", np.median(list(md_val_ratio.values())))
# print(pd.DataFrame(list(zip(list(md_val_ratio.keys()), list(md_val_ratio.values()))), columns=['Label', 'Ratio']))
print("    Training:Test :", np.median(list(md_test_ratio.values())))
# print(pd.DataFrame(list(zip(list(md_test_ratio.keys()), list(md_test_ratio.values()))), columns=['Label', 'Ratio']))


merged_val_ratio = (np.median(list(dd_val_ratio.values())) + np.median(list(md_val_ratio.values())))/2.0
merged_test_ratio = (np.median(list(dd_test_ratio.values())) + np.median(list(md_test_ratio.values())))/2.0

print()

print("~~~~ MERGED Target Ratios ~~~~")
print("    Training:Validation :", merged_val_ratio)
print("    Training:Test :", merged_test_ratio)

print()

merged_data = {}
for k in split_keys:
    merged_data[k] = pd.concat([dd[k], md[k]], ignore_index=True)

merged_counts = plot_data_distribution(merged_data, 'label', 'merged')

TOTAL_TRAIN_SENTENCES = 12000
TOTAL_VAL_SENTENCES = int(TOTAL_TRAIN_SENTENCES/merged_val_ratio)
TOTAL_TEST_SENTENCES = int(TOTAL_TRAIN_SENTENCES/merged_test_ratio)

def get_sentences(total_sentences, key="train"):
    global merged_data, merged_counts
    labels_not_considered = len(label_list)
    sent_in_each = total_sentences/labels_not_considered
    data = merged_data[key]
    count = dict(sorted(merged_counts[key].items(), key=lambda item: item[1]))

    sentences = {}

    for k, v in count.items():
        sentences[k] = min(v, sent_in_each)
        total_sentences -= sentences[k]
        labels_not_considered -= 1
        try:
            sent_in_each = int(total_sentences/labels_not_considered)
        except:
            pass

    return sentences

def generate_dataset(sent_count, key):
    global merged_data
    data = merged_data[key]

    list_of_df = []

    for i in label_list:
        temp = data[data["label"] == i].reset_index(drop=True)
        rows = np.random.choice(temp.index.values, sent_count[i], replace=False)
        list_of_df.append(temp.iloc[rows])

    return pd.concat(list_of_df, ignore_index=True)


train_sent_count = get_sentences(TOTAL_TRAIN_SENTENCES, "train")
val_sent_count = get_sentences(TOTAL_VAL_SENTENCES, "validation")
test_sent_count = get_sentences(TOTAL_TEST_SENTENCES, "test")


balanced_data = {}
balanced_data["train"] = generate_dataset(train_sent_count, "train")
balanced_data["validation"] = generate_dataset(val_sent_count, "validation")
balanced_data["test"] = generate_dataset(test_sent_count, "test")


plot_data_distribution(balanced_data, 'label', 'balanced')

for k in split_keys:
    balanced_data[k] = balanced_data[k].sample(frac=1, random_state=SEED).reset_index(drop=True)
    balanced_data[k].to_csv(ROOT+"data/balanced_emotion/"+k+'.txt', index=False, header=False, sep="_")

print("~~~~ GENERATED BALANCED EMO DATASET ~~~~")
print("Training Sentences: ", sum(list(train_sent_count.values())))
print("Validation Sentences: ", sum(list(val_sent_count.values())))
print("Test Sentences: ", sum(list(test_sent_count.values())))