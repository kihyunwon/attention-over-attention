import os
import time
import nltk
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

  words = []
  for line in input_stream:
    tokens = sentence_to_tokens_fn(line.rstrip('\n').rstrip('##########'))
    if len(tokens) > 0:
      tokens += ['<s>', '</s>', '<unk>', '<query_end>']
      words += [token.lower() for token in tokens if token.lower() not in cachedStopWords]
  
  vocab = corpora.Dictionary([words], prune_at=vocab_size)
  print(" [*] Tokenize : %.4fs" % (t0 - time.time()))

  return vocab

def add_vocabulary(vocab, input_stream, vocab_size, sentence_to_tokens_fn=None):
  new_vocab = create_vocabulary(input_stream, vocab_size, sentence_to_tokens_fn)
  return vocab.merge_with(new_vocab)

def save_vocabulary(vocab, vocabulary_file):
  vocab.save(vocabulary_path)

def load_vocabulary(vocabulary_file):
  if os.path.exists(vocabulary_path):
    vocab = corpora.Dictionary.load(vocabulary_path)
    return vocab
  else:
    raise ValueError(" [!] Vocabulary file %s not found.", vocabulary_path)
