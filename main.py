import pandas as pd
from flair.models import SequenceTagger
from flair.data import Sentence
import spacy
import os
import re


def setup():
    # load the POS tagger
    tagger = SequenceTagger.load("flair/pos-english")
    return tagger


class MyData:
    """
    Class for handling sentences from a csv file
    """

    def __init__(self, filename: str):
        """

        :param filename:
        """
        self.df = pd.read_csv(filename, skiprows=2, header=2)
        self.query_description = pd.read_csv(filename, skiprows=1, nrows=2).iloc[1].iloc[1]
        self.prompts = None
        self.joined_df = None
        self.tagged = []

    def find_sentences(self) -> None:

        self.joined_df = pd.DataFrame(columns=["sentence", "KWIC"])
        for row in self.df.iterrows():

            # get the KWIC
            row = row[1]
            text = row["KWIC"]

            # tags indicating sentences
            sent_beginning = "<s>"
            sent_end = "</s>"

            # find the beginning and the end of the sentence
            l_idx = row["Left"].rfind(sent_beginning)
            r_idx = row["Right"].find(sent_end)

            if l_idx == -1 or r_idx == -1:
                continue

            # get the sentence
            left = row["Left"][l_idx + len(sent_beginning):]
            right = row["Right"][:r_idx]
            text = left + " " + text + " " + right

            self.joined_df = pd.concat(
                [self.joined_df, pd.DataFrame(data=[[text, row["KWIC"]]], columns=["sentence", "KWIC"])],
                ignore_index=True)

    def tag(self):

        tagger = setup()
        for row in self.df.iterrows():
            text = row[1] + row[2] + row[3]
            print(text)
            # make a sentence
            sentence = Sentence(text)

            # predict NER tags
            tagger.predict(sentence)

            # print sentence with predicted tags
            print(sentence.to_tagged_string())

            # get ner chunks
            for entity in sentence.get_spans('pos'):
                self.tagged.append(entity)

    def compute_prompts(self):
        self.prompts = pd.DataFrame(columns=["sent", "prompt"])


def load_data(filenames: list[str]):
    MyData_lst = []
    for filename in filenames:
        MyData_lst.append(MyData(filename))
    return MyData_lst

