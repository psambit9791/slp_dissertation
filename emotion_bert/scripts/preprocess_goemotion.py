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

emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
id_emotion_map = {0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'}

ekman_emo_map = {"anger": "anger", "annoyance": "anger", "disapproval": "anger", 
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "happiness","amusement": "happiness","approval": "happiness","excitement": "happiness","gratitude": "happiness","love": "happiness","optimism": "happiness","relief": "happiness","pride": "happiness", "admiration": "happiness", "desire": "happiness", "caring": "happiness",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise", "curiosity": "surprise",
    "neutral": "neutral"
    }
id_ekman_mapping = {"anger": 2, "disgust": 6, "fear": 5, "happiness": 1, "neutral": 0, "sadness": 3, "surprise": 4}
emo_map = {2: "anger", 6: "disgust", 5: "fear", 1: "happiness", 0: "neutral", 3: "sadness", 4: "surprise"}

label_list = [0, 1, 2, 3, 4, 5, 6]

ge = load_dataset('go_emotions')
gedict = {}

def plot_data_distribution(data_dict, label_var, filename=None, mapper=emo_map):
    global split_keys, label_list
    counts = {}
    fig, ax = plt.subplots(1, 3, figsize=(26,12))

    for i, k in enumerate(split_keys):
        counts[k] = data_dict[k][label_var].value_counts().to_dict()
        for unkey in set(label_list).difference(set(counts[k].keys())):
            counts[k][unkey] = 0
        x = []
        y = []
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

def replace_labels(old_labels):
    global id_emotion_map, ekman_emo_map, id_ekman_mapping
    new_labels = []
    for idx,l in enumerate(old_labels):
        old_l = id_emotion_map[int(l)]
        new_l = ekman_emo_map[old_l]
        new_labels.append(str(id_ekman_mapping[new_l]))
    new_labels = list(set(new_labels))
    return new_labels

for i in split_keys:
    gedict[i] = pd.DataFrame(ge[i])


# plot_data_distribution(gedict, 'labels', "goemotion_27", id_emotion_map)

for i in split_keys:
    gedict[i] = pd.DataFrame(ge[i])
    gedict[i]["labels"] = gedict[i]["labels"].map(replace_labels)

# plot_data_distribution(gedict, 'labels', "unbalanced_goemotion")

def prioirity_emotion_selection(df_dict):

    def get_emotion(emolist):
        if len(emolist) == 1:
            return int(emolist[0])
        for i in ['6', '5', '3', '4', '2', '0', '1']:
            if i in emolist:
                return int(i)

    # Prioirity Order: 6, 5, 3, 4, 2, 1, 0
    for k in split_keys:
        df_dict[k]["labels"] = df_dict[k]["labels"].map(get_emotion)
    return df_dict


def sentence_count(data_dict):
    count = {}
    for i in split_keys:
        count[i] = {}
        for idx, row in data_dict[i].iterrows():
            try:
                if len(row["labels"]) == 1:
                    try:
                        count[i][row["labels"][0]] += 1
                    except KeyError:
                        count[i][row["labels"][0]] = 1
            except:
                try:
                    count[i][row["labels"]] += 1
                except KeyError:
                    count[i][row["labels"]] = 1

    return count

gedict = prioirity_emotion_selection(gedict)
plot_data_distribution(gedict, 'labels', "unbalanced_goemotion")
for k in split_keys:
    gedict[k].drop(columns=['id'], inplace=True)
    gedict[k].rename(columns={'labels': 'label'}, inplace=True)


# for k in split_keys:
#     gedict[k] = gedict[k].sample(frac=1, random_state=SEED).reset_index(drop=True)
#     gedict[k] = gedict[k][['label', 'text']]
#     gedict[k].to_csv(ROOT+"data/goemotion/"+k+'.txt', index=False, header=False, sep="\t")