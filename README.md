Attentive-over-Attention Neural Networks for Reading Comprehension
==================================================================

Tensorflow implementation of [Attentive-over-Attention Neural Networks](https://arxiv.org/abs/1607.04423).


Prerequisites
-------------

- Python 3.5+
- [Tensorflow](https://www.tensorflow.org/)
- [NLTK](http://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/index.html)


Usage
-----

First, download [DeepMind Q&A Dataset](https://github.com/deepmind/rc-data) from [here](http://cs.nyu.edu/~kcho/DMQA/), and untar `cnn.tgz` and `dailymail.tgz` into `data` directory:

Then run the pre-processing code with:

    $ ./prepare-rc.sh

To train a model with `cnn` dataset:

    $ python3 main.py --dataset cnn -t

To test an existing model (in progress):

    $ python3 main.py --dataset cnn


Credit
------

Modified codes for pre-processing, shuffling, and loading dataset are originally from IBM's [Attention Sum Reader](https://github.com/rkadlec/asreader/blob/master/asreader.git) implementation.


Results (in progress)
-------
