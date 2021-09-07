import os
from OtherUtils import load_vocab
import json
from typing import List
from pypinyin import pinyin, Style


class BertDataset(object):
    def __init__(self, bert_path, max_length: int = 512):
        super().__init__()
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        config_path = os.path.join(bert_path, 'config')
        self.max_length = max_length

        self.chardict = load_vocab(vocab_file)
        self.chardictlen = len(self.chardict)
        self.inversechardict = {v: k for k, v in self.chardict.items()}

        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def tokenize_sentence(self, sentence):
        # convert sentence to ids
        bert_tokens = [self.chardict["[CLS]"]] + \
                      [self.chardict[c] if c in self.chardict.keys() else self.chardict["[UNK]"] for c in sentence] + \
                      [self.chardict["[SEP]"]]

        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence)
        # assert，token nums should be same as pinyin token nums
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)

        return bert_tokens, pinyin_tokens

    def convert_sentence_to_pinyin_ids(self, sentence: str) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, errors=lambda x: [['not chinese'] for _ in x])

        pinyin_locs = [[0] * 8]
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]

            if pinyin_string in self.pinyin2tensor:
                pinyin_locs.append(self.pinyin2tensor[pinyin_string])
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs.append(ids)

        pinyin_locs.append([0] * 8)

        return pinyin_locs
