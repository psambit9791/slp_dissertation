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

MAX_SENTENCES = 38000

split_keys = ["train", "validation", "test"]
label_list = [0, 1, 2, 3, 4, 5, 6]

emo_map = {0: "anger", 1: "disgust" , 2: "fear", 3: "happiness", 4: "neutral", 5: "sadness", 6: "surprise", 7: "excitement", 8: "frustration", 9: "other", 10: "xxx"}
reverse_emo_map = {"anger": 0, "disgust": 1 , "fear": 2, "happiness": 3, "neutral": 4, "sadness": 5, "surprise": 6, "excitement": 7, "frustration": 8, "other": 9, "xxx": 10}

# To ensure removal of classes does not affect the model
# Larger number of examples = Lower value class (0 for max examples)
final_emo_map = {4: 0, 3: 1, 0: 2, 5: 3, 6: 4, 2: 5, 1: 6}
final_emo_lbl = {2: "anger", 6: "disgust", 5: "fear", 1: "happiness", 0: "neutral", 3: "sadness", 4: "surprise"}
final_reverse_emo_lbl = {"anger": 2, "disgust": 6, "fear": 5, "happiness": 1, "neutral": 0, "sadness": 3, "surprise": 4}

min_words = {"train": 4000, "validation": 400, "test": 300}

mixed = {}
mixed["train"] = pd.read_csv(ROOT+"data/balanced_emotion_all/train.txt", sep="\t", names=['label', 'text'])
mixed["validation"] = pd.read_csv(ROOT+"data/balanced_emotion_all/validation.txt", sep="\t", names=['label', 'text'])
mixed["test"] = pd.read_csv(ROOT+"data/balanced_emotion_all/test.txt", sep="\t", names=['label', 'text'])


goemo = {}
goemo["train"] = pd.read_csv(ROOT+"data/goemotion/train.txt", sep="\t", names=['label', 'text'])
goemo["validation"] = pd.read_csv(ROOT+"data/goemotion/validation.txt", sep="\t", names=['label', 'text'])
goemo["test"] = pd.read_csv(ROOT+"data/goemotion/test.txt", sep="\t", names=['label', 'text'])


def plot_data_distribution(data_dict, label_var, filename=None, mapper=final_emo_lbl):
    global split_keys, label_list
    counts = {}
    fig, ax = plt.subplots(1, 3, figsize=(26,12))

    for i, k in enumerate(split_keys):
        counts[k] = data_dict[k][label_var].value_counts().to_dict()
        for unkey in set(label_list).difference(set(counts[k].keys())):
            counts[k][unkey] = 0
        x = []
        y = []
        print(counts[k])
        for j in sorted(list(counts[k].keys())):
            x.append(mapper[j])
            y.append(counts[k][j])
        ax[i].bar(x, y)
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

mixed_counts = plot_data_distribution(mixed, 'label', 'mixed_init')
goemo_counts = plot_data_distribution(goemo, 'label', 'goemo_init')
# merged_counts = plot_data_distribution(merged, 'label', 'merged_init')


def get_ratios(counts):

    tr_val = {}
    tr_ts = {}

    for lbl in label_list:
        tr_val[lbl] = counts["train"][lbl]/counts["validation"][lbl]
        tr_ts[lbl] = counts["train"][lbl]/counts["test"][lbl]
    
    return tr_val, tr_ts

mixed_val_ratio, mixed_test_ratio = get_ratios(mixed_counts)
print("~~~~ MIXED Ratios ~~~~")
print("    Training:Validation :", np.median(list(mixed_val_ratio.values())))
# print(pd.DataFrame(list(zip(list(dd_val_ratio.keys()), list(dd_val_ratio.values()))), columns=['Label', 'Ratio']))
print("    Training:Test :", np.median(list(mixed_test_ratio.values())))
# print(pd.DataFrame(list(zip(list(dd_test_ratio.keys()), list(dd_test_ratio.values()))), columns=['Label', 'Ratio']))

print()

goemo_val_ratio, goemo_test_ratio = get_ratios(goemo_counts)
print("~~~~ GOEMO Ratios ~~~~")
print("    Training:Validation :", np.median(list(goemo_val_ratio.values())))
# print(pd.DataFrame(list(zip(list(md_val_ratio.keys()), list(md_val_ratio.values()))), columns=['Label', 'Ratio']))
print("    Training:Test :", np.median(list(goemo_test_ratio.values())))
# print(pd.DataFrame(list(zip(list(md_test_ratio.keys()), list(md_test_ratio.values()))), columns=['Label', 'Ratio']))


merged_val_ratio = (np.median(list(mixed_val_ratio.values())) + np.median(list(goemo_val_ratio.values())))/2.0
merged_test_ratio = (np.median(list(mixed_test_ratio.values())) + np.median(list(goemo_test_ratio.values())))/2.0

print()

print("~~~~ MERGED Target Ratios ~~~~")
print("    Training:Validation :", merged_val_ratio)
print("    Training:Test :", merged_test_ratio)

print()

merged_data = {}
for k in split_keys:
    merged_data[k] = pd.concat([mixed[k], goemo[k]], ignore_index=True)


def drop_column(data_dict, emo):
    global final_reverse_emo_map
    emo_codes = []
    for e in emo:
        emo_codes.append(final_reverse_emo_lbl[e])
    for k in split_keys:
        for emo_code in emo_codes:
            data_dict[k] = data_dict[k].drop(data_dict[k][data_dict[k].label == emo_code].index)
        data_dict[k].reset_index(inplace=True)
        data_dict[k].drop(columns=["index"], inplace=True)
    return data_dict

# print(merged_data["test"])
# merged_data = drop_column(merged_data, ["fear", "disgust"])

label_list = merged_data["train"]["label"].unique()
# print(merged_data["test"])
merged_counts = plot_data_distribution(merged_data, 'label', 'merged')

print("TOTAL SENTENCES AFTER MERGING:\n", merged_counts)

TOTAL_TRAIN_SENTENCES = MAX_SENTENCES
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

# balanced_data = drop_column(["fear", "disgust"])

plot_data_distribution(balanced_data, 'label', 'final', final_emo_lbl)

for k in split_keys:
    balanced_data[k] = balanced_data[k].sample(frac=1, random_state=SEED).reset_index(drop=True)
    balanced_data[k].to_csv(ROOT+"data/final_dataset/"+k+'.txt', index=False, header=False, sep="\t")

print("~~~~ GENERATED BALANCED EMO DATASET ~~~~")
print("Training Sentences: ", sum(list(train_sent_count.values())))
print("Validation Sentences: ", sum(list(val_sent_count.values())))
print("Test Sentences: ", sum(list(test_sent_count.values())))