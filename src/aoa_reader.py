import time
import numpy as np
import tensorflow as tf

from .reader import Reader

"""
Tensorflow implementation of the Attention-over-Attention Neural Networks for Reading Comprehension
https://arxiv.org/abs/1607.04423
"""


class AoAReader(Reader):
  
  def __init__(self, args):
    super(AoAReader, self).__init__(args)

    self.d = self.args.document_size
    self.q = self.args.query_size
    self.max_nsteps = self.d + self.q

    print(" [*] Building Bidirectional GRU layer...")
    self.fw_cell = tf.nn.rnn_cell.GRUCell(self.args.hidden_dims)
    self.bw_cell = tf.nn.rnn_cell.GRUCell(self.args.hidden_dims)

    self.initial_state_fw = self.fw_cell.zero_state(self.args.batch_size, tf.float32)
    self.initial_state_bw = self.bw_cell.zero_state(self.args.batch_size, tf.float32)

  def prepare_model(self, vocab_dict):
    self.vocab_size = len(vocab_dict)

    # constant
    vocab = tf.constant(list(vocab_dict.values()))

    self.emb = tf.get_variable("emb", [self.vocab_size, self.args.embedding_dims])
    if self.args.train and self.args.keep_prob < 1:
      self.emb = tf.nn.dropout(self.emb, self.args.keep_prob)

    # inputs
    self.inputs = tf.placeholder(tf.int32, [self.args.batch_size, self.max_nsteps])
    embed_inputs = tf.nn.embedding_lookup(self.emb, self.inputs)

    tf.histogram_summary("embed", self.emb)

    # sequence length
    _seq_len = tf.fill(tf.expand_dims(self.args.batch_size, 0),
                       tf.constant(self.max_nsteps, dtype=tf.int64))

    # outputs, states
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
                        self.fw_cell,
                        self.bw_cell,
                        embed_inputs,
                        sequence_length=_seq_len,
                        initial_state_fw=self.initial_state_fw,
                        initial_state_bw=self.initial_state_bw,
                        dtype=tf.float32)

    # concat output
    outputs = tf.concat(2, outputs)

    # select document & query
    d = outputs[:,:self.d,:]
    q = outputs[:,self.d:,:]

    # batch pair-wise matching
    i_att = tf.batch_matmul(d, q, adj_y=True) # shape = (batch_size, self.d, self.q)
    # individual attentions
    alpha = tf.map_fn(lambda x: tf.nn.softmax(tf.transpose(x)), i_att)
    # attention-over-attentions
    beta_t = tf.map_fn(tf.nn.softmax, i_att)
    beta = tf.map_fn(lambda x: tf.reduce_mean(x, 0), beta_t) # shape = (batch_size, self.q, )
    beta = tf.reshape(beta, [self.args.batch_size, self.q, 1])
    # document-level attention
    s = tf.batch_matmul(alpha, beta, adj_x=True) # shape = (batch_size, self.d, 1)

    document = self.inputs[:,:self.d]

    # TODO: optimize this?
    def predict(pos):
      # init empty matrix
      init = tf.zeros(shape=(self.args.batch_size, self.d))
      # for each word in vocab, we set i,j
      # where i,j is a position of the word in the document
      for i in range(self.args.batch_size):
        for j in range(self.d):
          if document[i,j] == pos:
            init[i,j] = 1
      return init

    # mask
    mask = tf.map_fn(predict, vocab, dtype=tf.float32)
    mask = tf.reshape(mask, [self.args.batch_size, self.vocab_size, self.d])

    # prediction
    self.y_ = tf.reshape(tf.batch_matmul(mask, s), [self.args.batch_size, self.vocab_size])
    tf.histogram_summary("output", self.y_)

    # answer
    self.y = tf.placeholder(tf.float32, [self.args.batch_size, self.vocab_size])

    self.loss = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.y)
    tf.scalar_summary("loss", tf.reduce_mean(self.loss))

    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", self.accuracy)

    print(" [*] Model built.")

  def train(self, sess, vocab_size, epoch=25, learning_rate=1e-4,
            beta1=0.9, beta2=0.95, data_dir="data", data_type="cnn"):

    # load data
    dataset, vocab = self.get_dataset("training.txt")
    # build model
    self.prepare_model(vocab.token2id)

    # gradient clipping
    self.op = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
    gvs = self.op.compute_gradients(self.loss)
    capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
    self.op.apply_gradients(gvs)

    sess.run(tf.initialize_all_variables())

    if self.load(sess, self.args.checkpoint_dir, data_type):
      print(" [*] Model checkpoint is loaded.")
    else:
      print(" [*] There is no checkpoint for this model.")
    
    y = np.zeros([self.args.batch_size, vocab_size])

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/aoa_reader", sess.graph)

    counter = 0
    start_time = time.time()
    for epoch_idx in range(epoch):
      # data loader
      batch = dataset.batch_loader()

      batch_stop = False
      while True:
        y.fill(0)
        batch_idx = 0
        inputs, answers = [], []
        while True:
          try:
            context, question, answer, _ = next(batch)
          except StopIteration:
            batch_stop = True
            break

          data = context.append(question)
          inputs.append(data)
          y[batch_idx][int(answer[0])] = 1

          batch_idx += 1
          if batch_idx == self.args.batch_size: break
        if batch_stop: break

        _, summary_str, cost, accuracy = sess.run([self.op, merged, self.loss, self.accuracy], 
                                                   feed_dict={self.inputs: inputs,
                                                              self.y: y})
        if counter % 10 == 0:
          writer.add_summary(summary_str, counter)
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
              % (epoch_idx, data_idx, self.args.batch_size, time.time() - start_time, np.mean(cost), accuracy))
        counter += 1
      
      self.save(sess, self.args.checkpoint_dir, data_type)

  #def test(self, sess, ckpt_dir, vocab_size, data_dir, data_type):
    #pass
