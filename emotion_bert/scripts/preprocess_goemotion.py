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
    gedict[i]["labels"] = gedict[i]["labels"].map(replace_labels)

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
plot_data_distribution(gedict, 'labels', "goemotion")
for k in split_keys:
    gedict[k].drop(columns=['id'], inplace=True)
    gedict[k].rename(columns={'labels': 'label'}, inplace=True)


for k in split_keys:
    gedict[k] = gedict[k].sample(frac=1, random_state=SEED).reset_index(drop=True)
    gedict[k].to_csv(ROOT+"data/goemotion/"+k+'.txt', index=False, header=False, sep="\t")

# def plot_data_distribution(data_dict, label_var, filename=None, mapper=emo_map):
#     global split_keys, label_list
#     counts = {}
#     fig, ax = plt.subplots(1, 3, figsize=(26,12))

#     for i, k in enumerate(split_keys):
#         counts[k] = data_dict[k][label_var].value_counts().to_dict()
#         for unkey in set(label_list).difference(set(counts[k].keys())):
#             counts[k][unkey] = 0
#         x = []
#         y = []
#         for j in sorted(list(counts[k].keys())):
#             x.append(mapper[j])
#             y.append(counts[k][j])
#         ax[i].bar(x, y)
#         ax[i].set_ylabel("Number of occurences")
#         ax[i].set_xlabel("Emotion")
#         ax[i].set_title("Data for " + str(k))

    
#     if filename is not None:
#         plt.tight_layout()
#         plt.savefig(ROOT+"images/"+filename+".png")
#         plt.clf()
#     else:
#         plt.show()

#     return counts


# def clean_dyda():
#     global dd, dd_counts, min_words, split_keys
#     for k in split_keys:
#         emotions_with_large_data = []
#         for emo in label_list:
#             if dd_counts[k][emo] > min_words[k]:
#                 emotions_with_large_data.append(emo)
#         to_delete = []
#         for idx, row in dd[k].iterrows():
#             if row["Label"] in emotions_with_large_data and row['Num_Tokens'] <= 4:
#                 to_delete.append(idx)
#         dd[k].drop(to_delete, inplace=True)
#         dd[k].drop(columns=['Dialogue_ID', 'Idx', 'Emotion', 'Num_Tokens'], inplace=True)
#         dd[k].rename(columns={'Utterance': 'text', 'Label': 'label'}, inplace=True)
#     return dd

# def clean_meld():
#     global md, md_counts, min_words, split_keys
#     for k in split_keys:
#         emotions_with_large_data = []
#         for emo in label_list:
#             if md_counts[k][emo] > min_words[k]:
#                 emotions_with_large_data.append(emo)
#         to_delete = []
#         for idx, row in md[k].iterrows():
#             if row["Label"] in emotions_with_large_data and row['Num_Tokens'] <= 4:
#                 to_delete.append(idx)
#         md[k].drop(to_delete, inplace=True)
#         md[k].drop(columns=['Speaker', 'Dialogue_ID', 'Utterance_ID', 'Emotion', 'Idx', 'Num_Tokens'], inplace=True)
#         md[k].rename(columns={'Utterance': 'text', 'Label': 'label'}, inplace=True)
#     return md

# def clean_iemocap():
#     global iemo, iemo_counts, min_words, split_keys
#     to_exclude = {'fru', 'xxx', 'exc', 'oth'}
#     for k in split_keys:
#         to_delete = []
#         for emo in to_exclude:
#             to_delete += np.where(iemo[k]['Emotion'] == emo)[0].tolist()
#         emotions_with_large_data = []
#         for emo in label_list:
#             if iemo_counts[k][emo] > min_words[k]:
#                 emotions_with_large_data.append(emo)
#         for idx, row in iemo[k].iterrows():
#             if row["Label"] in emotions_with_large_data and row['Num_Tokens'] <= 4:
#                 to_delete.append(idx)
#         to_delete = list(set(to_delete))

#         iemo[k].drop(to_delete, inplace=True)
#         iemo[k].drop(columns=['Dialogue_ID', 'Utterance_ID', 'Emotion', 'Idx', 'Num_Tokens'], inplace=True)
#         iemo[k].rename(columns={'Utterance': 'text', 'Label': 'label'}, inplace=True)
#     return iemo

# def standardize_labels(df, mapper):
#     for emo in set(list(mapper.keys())):
#         df.loc[df.Emotion == emo, "Label"] = mapper[emo]
#     return df

# def add_token_length(df):
#     lengths = []
#     for idx, row in df.iterrows():
#         words= [word for word in word_tokenize(str(row['Utterance'])) if word.isalnum()]
#         lengths.append(len(words))
#     df['Num_Tokens'] = lengths
#     return df

# iemo_map = {'sad': 5, 'fru': 8, 'xxx': 10, 'ang': 0, 'neu': 4, 'exc': 7, 'hap': 3, 'sur': 6, 'dis': 1, 'fea': 2, 'oth': 9}

# for i in split_keys:
#     dd[i] = pd.DataFrame(dyda_e[i])
#     dd[i] = add_token_length(dd[i])
#     md[i] = pd.DataFrame(meld_e[i])
#     md[i] = add_token_length(md[i])
#     iemo[i] = pd.DataFrame(iemocap[i])
#     iemo[i] = add_token_length(iemo[i])
#     iemo[i] = standardize_labels(iemo[i], iemo_map)


# dd_counts = plot_data_distribution(dd, 'Label', 'dyda_init')
# md_counts = plot_data_distribution(md, 'Label', 'meld_init')
# iemo_counts = plot_data_distribution(iemo, 'Label', 'iemo_init')

# clean_dyda()
# clean_meld()
# clean_iemocap()

# dd_counts = plot_data_distribution(dd, 'label', 'dyda_cleaned')
# md_counts = plot_data_distribution(md, 'label', 'meld_cleaned')
# iemo_counts = plot_data_distribution(iemo, 'label', 'iemo_cleaned')


# def get_ratios(counts):

#     tr_val = {}
#     tr_ts = {}

#     for lbl in label_list:
#         tr_val[lbl] = counts["train"][lbl]/counts["validation"][lbl]
#         tr_ts[lbl] = counts["train"][lbl]/counts["test"][lbl]
    
#     return tr_val, tr_ts

# dd_val_ratio, dd_test_ratio = get_ratios(dd_counts)
# print("~~~~ DYDA Ratios ~~~~")
# print("    Training:Validation :", np.median(list(dd_val_ratio.values())))
# # print(pd.DataFrame(list(zip(list(dd_val_ratio.keys()), list(dd_val_ratio.values()))), columns=['Label', 'Ratio']))
# print("    Training:Test :", np.median(list(dd_test_ratio.values())))
# # print(pd.DataFrame(list(zip(list(dd_test_ratio.keys()), list(dd_test_ratio.values()))), columns=['Label', 'Ratio']))

# print()

# md_val_ratio, md_test_ratio = get_ratios(md_counts)
# print("~~~~ MELD Ratios ~~~~")
# print("    Training:Validation :", np.median(list(md_val_ratio.values())))
# # print(pd.DataFrame(list(zip(list(md_val_ratio.keys()), list(md_val_ratio.values()))), columns=['Label', 'Ratio']))
# print("    Training:Test :", np.median(list(md_test_ratio.values())))
# # print(pd.DataFrame(list(zip(list(md_test_ratio.keys()), list(md_test_ratio.values()))), columns=['Label', 'Ratio']))

# print()

# iemo_val_ratio, iemo_test_ratio = get_ratios(iemo_counts)
# print("~~~~ IEMO Ratios ~~~~")
# print("    Training:Validation :", np.median(list(iemo_val_ratio.values())))
# # print(pd.DataFrame(list(zip(list(md_val_ratio.keys()), list(md_val_ratio.values()))), columns=['Label', 'Ratio']))
# print("    Training:Test :", np.median(list(iemo_test_ratio.values())))
# # print(pd.DataFrame(list(zip(list(md_test_ratio.keys()), list(md_test_ratio.values()))), columns=['Label', 'Ratio']))


# merged_val_ratio = (np.median(list(dd_val_ratio.values())) + np.median(list(md_val_ratio.values())) + np.median(list(iemo_val_ratio.values())))/3.0
# merged_test_ratio = (np.median(list(dd_test_ratio.values())) + np.median(list(md_test_ratio.values())) + np.median(list(iemo_test_ratio.values())))/3.0

# print()

# print("~~~~ MERGED Target Ratios ~~~~")
# print("    Training:Validation :", merged_val_ratio)
# print("    Training:Test :", merged_test_ratio)

# print()

# merged_data = {}
# for k in split_keys:
#     merged_data[k] = pd.concat([dd[k], md[k], iemo[k]], ignore_index=True)


# def reset_emotion_labels(df, mapper):

#     def new_lbl(item):
#         global final_emo_map
#         return final_emo_map[item]

#     for k in split_keys:
#         df[k]["label"] = df[k]["label"].map(new_lbl)
#     return df

# def drop_column(data_dict, emo):
#     global final_reverse_emo_map
#     emo_codes = []
#     for e in emo:
#         emo_codes.append(final_reverse_emo_lbl[e])
#     for k in split_keys:
#         for emo_code in emo_codes:
#             data_dict[k] = data_dict[k].drop(data_dict[k][data_dict[k].label == emo_code].index)
#         data_dict[k].reset_index(inplace=True)
#         data_dict[k].drop(columns=["index"], inplace=True)
#     return data_dict

# # print(merged_data["test"])
# merged_data = reset_emotion_labels(merged_data, final_emo_map)
# merged_data = drop_column(merged_data, ["fear", "disgust"])

# label_list = merged_data["train"]["label"].unique()
# # print(merged_data["test"])
# merged_counts = plot_data_distribution(merged_data, 'label', 'merged')

# print("TOTAL SENTENCES AFTER MERGING:\n", merged_counts)

# TOTAL_TRAIN_SENTENCES = MAX_SENTENCES
# TOTAL_VAL_SENTENCES = int(TOTAL_TRAIN_SENTENCES/merged_val_ratio)
# TOTAL_TEST_SENTENCES = int(TOTAL_TRAIN_SENTENCES/merged_test_ratio)

# def get_sentences(total_sentences, key="train"):
#     global merged_data, merged_counts
#     labels_not_considered = len(label_list)
#     sent_in_each = total_sentences/labels_not_considered
#     data = merged_data[key]
#     count = dict(sorted(merged_counts[key].items(), key=lambda item: item[1]))

#     sentences = {}

#     for k, v in count.items():
#         sentences[k] = min(v, sent_in_each)
#         total_sentences -= sentences[k]
#         labels_not_considered -= 1
#         try:
#             sent_in_each = int(total_sentences/labels_not_considered)
#         except:
#             pass

#     return sentences

# def generate_dataset(sent_count, key):
#     global merged_data
#     data = merged_data[key]

#     list_of_df = []

#     for i in label_list:
#         temp = data[data["label"] == i].reset_index(drop=True)
#         rows = np.random.choice(temp.index.values, sent_count[i], replace=False)
#         list_of_df.append(temp.iloc[rows])

#     return pd.concat(list_of_df, ignore_index=True)

# train_sent_count = get_sentences(TOTAL_TRAIN_SENTENCES, "train")
# val_sent_count = get_sentences(TOTAL_VAL_SENTENCES, "validation")
# test_sent_count = get_sentences(TOTAL_TEST_SENTENCES, "test")

# balanced_data = {}
# balanced_data["train"] = generate_dataset(train_sent_count, "train")
# balanced_data["validation"] = generate_dataset(val_sent_count, "validation")
# balanced_data["test"] = generate_dataset(test_sent_count, "test")

# # balanced_data = drop_column(["fear", "disgust"])

# plot_data_distribution(balanced_data, 'label', 'balanced', final_emo_lbl)

# for k in split_keys:
#     balanced_data[k] = balanced_data[k].sample(frac=1, random_state=SEED).reset_index(drop=True)
#     balanced_data[k].to_csv(ROOT+"data/balanced_emotion/"+k+'.txt', index=False, header=False, sep="_")

# print("~~~~ GENERATED BALANCED EMO DATASET ~~~~")
# print("Training Sentences: ", sum(list(train_sent_count.values())))
# print("Validation Sentences: ", sum(list(val_sent_count.values())))
# print("Test Sentences: ", sum(list(test_sent_count.values())))