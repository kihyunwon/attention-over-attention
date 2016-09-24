# Modified version of https://github.com/rkadlec/asreader/blob/master/asreader/text_comprehension/datasets/cnn_dataset.py

import numpy as np
from gensim import corpora
from picklable_itertools import iter_, chain

from . import cnn_ne_shuffle


'''
    This module serves for reading the CNN and Daily Mail datasets as introduced in http://arxiv.org/abs/1603.01547
    into the ASReader model.
'''

class CNNDataset:
    def __init__(self, files, dictionary, batch_size, document_size, query_size,
                 bos_token='<s>', eos_token='</s>', unk_token='<unk>', level='word',
                 preprocess=None, append_question=False, question_end_token='<query_end>',
                 add_attention_features=False):

        self.files = files
        if isinstance(dictionary, corpora.Dictionary):
            self.dictionary = dictionary.token2id
        else:
            self.dictionary = dictionary
        l = len(dictionary)
        markers = {'<s>': l, '</s>': l+1, '<unk>': l+2, '<query_end>': l+3}
        self.dictionary.update(markers)
        self.batch_size = batch_size
        self.d = document_size
        self.q = query_size

        if bos_token is not None and bos_token not in self.dictionary:
            raise ValueError
        self.bos_token = bos_token
        if eos_token is not None and eos_token not in self.dictionary:
            raise ValueError
        self.eos_token = eos_token
        if unk_token not in self.dictionary:
            raise ValueError
        self.unk_token = unk_token
        if level not in ('word', 'character'):
            raise ValueError

        self.level = level
        self.preprocess = preprocess
        self.append_question = append_question
        self.question_end_token = question_end_token
        self.add_attention_features = add_attention_features

        # set shuffle dictionary
        cnn_ne_shuffle.set_dictionary(self.dictionary)
        # load data
        self.data = self._process_data(state=self._open())

    def _open(self):
        return chain(*[iter_(open(f)) for f in self.files])

    def _translate_one_line(self, sentence, limit=-1):

        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data.extend(self.dictionary.get(word,
                                            self.dictionary[self.unk_token])
                        for word in sentence.split())
        else:
            data.extend(self.dictionary.get(char,
                                            self.dictionary[self.unk_token])
                        for char in sentence.strip())
        if self.eos_token:
            data.append(self.dictionary[self.eos_token])

        # padding
        if limit != -1:
            if limit == 1:
                data = data[2]
            elif limit <= len(data):
                data = data[:limit-1]
                data.append(self.dictionary[self.eos_token])
            else: # limit > len(data)
                for _ in range(limit - len(data)):
                    data.append(self.dictionary[self.eos_token])

        return np.array(data)

    def _process_story(self, lines):
        story = lines[1]    # Context document
        question = lines[2] # Query
        answer = lines[3]   # Correct answer
        candidates_list = lines[4:] # Answer candidates

        # Move the correct answer to the first position among the candidates
        replace = None
        for candidate in candidates_list:
            if answer in candidate:
                replace = candidate
                break

        if replace:
            candidates_list.remove(replace)
            candidates_list.insert(0, replace)

        candidates_strs = " ".join(candidates_list)

         # Add the question at the beginning and end of the context document to direct the context encoder
        if self.append_question:
            story = question + " " + self.question_end_token + " " + story + " " + self.question_end_token + " " + question

        return (self._translate_one_line(story, self.d), self._translate_one_line(question, self.q),
                self._translate_one_line(answer, 1), self._translate_one_line(candidates_strs),)

    def _process_data(self, state=None):
        data = []
        lines = []

        while True:
            try:
                line = next(state).strip()
                if line != "##########" and len(line) > 0:
                    lines.append(line)
                elif line == "##########":
                    data.append(self._process_story(lines))
                    lines = []
            except StopIteration:
                break

        return np.array(data)

    def batch_loader(self):
        # randomly shuffle
        np.random.shuffle(self.data)
        # select batch_size data
        batch = self.data[np.random.choice(len(self.data), size=self.batch_size, replace=False),:]
        # shuffle named entities
        for i in range(self.batch_size):
            yield cnn_ne_shuffle.shuffle_ne(batch[i])
