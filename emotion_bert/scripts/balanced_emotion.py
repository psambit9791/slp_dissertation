"""DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset"""
"""MELD: Multimodal EmotionLines Dataset"""


import os
from zipfile import ZipFile

import datasets
from datasets.tasks import TextClassification


_DESCRIPTION = """\
We develop a relatively balanced dataset merging the DailyDialog dataset and the MELD dataset. 
Training Sentences:  12000
Validation Sentences:  1264
Test Sentences:  1823
"""

# _URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"

emotion_label = {
    0: "no emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


class BalancedEmotion(datasets.GeneratorBasedBuilder):
    """DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset"""

    VERSION = datasets.Version("1.0.0")

    __SEP__ = "_"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=list(emotion_label.keys())),
                }
            ),
            supervised_keys=None,
            citation=None,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        # dl_dir = dl_manager.download_and_extract(_URL)
        dl_dir = "../data"
        data_dir = os.path.join(dl_dir, "balanced_emotion")

        # The splits are nested inside the zip
        # for name in ("train", "validation", "test"):
        #     zip_fpath = os.path.join(data_dir, f"{name}.zip")
        #     with ZipFile(zip_fpath) as zip_file:
        #         zip_file.extractall(path=data_dir)
        #         zip_file.close()

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "train.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "test.txt"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "validation.txt"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, file_path, split):
        """Yields examples."""
        # Yields (key, example) tuples from the dataset
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line_f in enumerate(f):
                if len(line_f.strip()) == 0:
                    break
                emotion, dialog = line_f.split(self.__SEP__)

                yield f"{split}-{i}", {
                    "text": dialog,
                    "label": emotion,
                }