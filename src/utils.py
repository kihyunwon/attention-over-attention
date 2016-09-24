import os
import time
import nltk
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords


cachedStopWords = stopwords.words("english")


def default_sentence_to_tokens(text):
  tokens = nltk.word_tokenize(text)
  return tokens

def create_vocabulary(input_stream, vocab_size, sentence_to_tokens_fn=None):
  t0 = time.time()
  print(" [*] Creating a new vocabulary...")

  if not sentence_to_tokens_fn:
    sentence_to_tokens_fn = default_sentence_to_tokens

  docs = []
  lines = []
  for line in input_stream:
    rline = line.strip()
    tokens = sentence_to_tokens_fn(rline)
    if '##########' not in tokens and len(rline) > 0:
      lines += [token.lower() for token in tokens if token.lower() not in cachedStopWords]
    elif '##########' in tokens:
      docs.append(lines)
      lines = []

  limit = np.abs(vocab_size - 4)
  vocab = corpora.Dictionary(docs)
  vocab.filter_extremes(no_below=1, no_above=0.7, keep_n=limit)
  print(" [*] Tokenize : %.4fs" % (time.time() - t0))

  return vocab

def add_vocabulary(vocab, input_stream, vocab_size, sentence_to_tokens_fn=None):
  if not sentence_to_tokens_fn:
    sentence_to_tokens_fn = default_sentence_to_tokens

  docs = []
  lines = []
  for line in input_stream:
    rline = line.strip()
    tokens = sentence_to_tokens_fn(rline)
    if '##########' not in tokens and len(rline) > 0:
      lines += [token.lower() for token in tokens if token.lower() not in cachedStopWords]
    elif '##########' in tokens:
      docs.append(lines)
      lines = []
  
  limit = np.abs(vocab_size - 4)
  vocab.add_documents(docs)
  vocab.filter_extremes(no_below=1, no_above=0.7, keep_n=limit)
  return vocab

def save_vocabulary(vocab, vocabulary_file):
  vocab.save(vocabulary_file)

def load_vocabulary(vocabulary_file):
  if os.path.exists(vocabulary_file):
    vocab = corpora.Dictionary.load(vocabulary_file)
    return vocab
  else:
    raise ValueError(" [!] Vocabulary file %s not found.", vocabulary_file)
