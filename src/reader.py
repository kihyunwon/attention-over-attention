import os
import re
import codecs
import numpy as np
import tensorflow as tf

from .cnn_dataset import CNNDataset 
from .utils import create_vocabulary, add_vocabulary, save_vocabulary, load_vocabulary

"""
Abstract Reader class.
"""


class Reader:
  def __init__(self, args):
    self.args = args
    self.vocab = None

  def save(self, sess, checkpoint_dir, data_type, global_step=None):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__ or "Reader"
    if self.args.batch_size:
      model_dir = "%s_%s_%s" % (model_name, data_type, self.args.batch_size)
    else:
      model_dir = data_type

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(sess, 
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def load(self, sess, checkpoint_dir, data_type):
    model_name = type(self).__name__ or "Reader"
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    if self.args.batch_size:
      model_dir = "%s_%s_%s" % (model_name, data_type, self.args.batch_size)
    else:
      model_dir = data_type
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False

  # Modified version of get_stream(...) found in
  # https://github.com/rkadlec/asreader/blob/master/asreader/text_comprehension/text_comprehension_base.py
  def get_dataset(self, file_type, vocab=None, add_dict=False, save=True):
    vocab_size = self.args.vocab_size
    data_dir = self.args.data_dir
    data_type = self.args.dataset
    batch_size = self.args.batch_size
    document_size = self.args.document_size
    query_size = self.args.query_size

    # Pattern for text tokenization
    pattern = re.compile(" |\t|\|")

    if data_type == "cnn" or data_type == "dailymail":
      prepro = lambda x : pattern.split(x)
    else:
      raise Exception(" [!] unsupported dataset")

    if file_type not in ("training.txt", "validation.txt", "test.txt"):
      raise Exception(" [!] unsupported file")

    t_file = os.path.join(data_dir, data_type, file_type)
    v_file = os.path.join(data_dir, data_type, "vocab_%d" % vocab_size)

    if not vocab:

      if os.path.exists(v_file):
        # load existing vocabulary
        vocab = load_vocabulary(v_file)
      else:
        print(" [*] Computing new vocabulary for file {}.".format(t_file))
        # compute vocabulary
        f = codecs.open(t_file, 'r', encoding="utf8")
        vocab = create_vocabulary(f, vocab_size, prepro)
        if save:
          save_vocabulary(vocab, v_file)

    if add_dict:
      # add words to vocab
      f = codecs.open(t_file, 'r', encoding="utf8")
      vocab = add_vocabulary(vocab, f, vocab_size, prepro)
      
      print(" [*] Added {} new words from file {} to previous vocabulary.".format(new_word_count, file))
      if save:
        save_vocabulary(vocab, v_file)

    # Select the data loader appropriate for the dataset
    common_params = { 'level': 'word', 'batch_size': batch_size, 
                      'document_size': document_size, 'query_size': query_size }

    print(" [*] Creating dataset...")
    return CNNDataset([t_file], vocab, **common_params), vocab
