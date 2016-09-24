import os
import argparse
import numpy as np
import tensorflow as tf

from src import AoAReader


parser = argparse.ArgumentParser(description="Inputs for AoAReader.")
parser.add_argument('-e', '--epoch', type=int, default="25",
                    help="Epoch to train [25]")
parser.add_argument('-vs', '--vocab_size', type=int, default="1000",
                    help="The size of vocabulary [10000]")
parser.add_argument('-bs', '--batch_size', type=int, default="32",
                    help="Batch size [32]")
parser.add_argument('-d', '--document_size', type=int, default="90",
                    help="Max length of document [90]")
parser.add_argument('-q', '--query_size', type=int, default="10",
                    help="Max length of query [10]")
parser.add_argument('-s', '--seed', type=int, default="2016",
                    help="Seed to set for random shuffle [2016]")
parser.add_argument('-emb', '--embedding_dims', type=int, default="384",
                    help="Dimensions of embeddings [384]")
parser.add_argument('-hid', '--hidden_dims', type=int, default="256",
                    help="Dimensions of GRU Cell [256]")
parser.add_argument('-lr', '--learning_rate', type=float, default="1e-4",
                    help="Learning rate [1e-4]")
parser.add_argument('-p', '--keep_prob', type=float, default="0.9",
                    help="1 - Dropout rate [0.9]")
parser.add_argument('-b1', '--beta1', type=float, default="0.9",
                    help="Beta1 for AdamOptimizer [0.9]")
parser.add_argument('-b2', '--beta2', type=float, default="0.95",
                    help="Beta2 for AdamOptimizer [0.95]")
parser.add_argument('-dr', '--data_dir', type=str, default="data",
                    help="The name of data directory [data]")
parser.add_argument('-ds', '--dataset', type=str, default="cnn",
                    help="The name of dataset to use [cnn]")
parser.add_argument('-ckpt', '--checkpoint_dir', type=str, default="checkpoint",
                    help="The name of directory to save the checkpoints [checkpoint]")
parser.add_argument('-t', '--train', action='store_true', default=False,
                    help="True for training, False for testing [False]")
args = parser.parse_args()

np.random.seed(args.seed)


def main(_):

  if not os.path.exists(args.checkpoint_dir):
    print(" [*] Creating checkpoint directory...")
    os.makedirs(args.checkpoint_dir)

  with tf.device('/cpu:0'), tf.Session() as sess:
    model = AoAReader(args)

    if args.train:
      model.train(sess, args.vocab_size, args.epoch,
                  args.learning_rate, args.beta1, args.beta2,
                  args.data_dir, args.dataset)
    else: # TODO: define test
      raise Exception(" [!] Test method undefined!")
      #model.test(sess, args.checkpoint_dir, args.vocab_size, args.data_dir, args.dataset)


if __name__ == '__main__':
  tf.app.run()
