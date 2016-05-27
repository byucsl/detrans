import sys
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import random

# globals
lstm_size = 8
number_of_layers = 2
batch_size = 5
n_steps = 5 # number of time steps
n_input = 3 # size of something?
n_hidden = 8 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)
learning_rate = 0.001
num_instances = 10

# fake test data
data = [ [ [ float( random.getrandbits( 1 ) ) for x in range( n_input ) ] for x in range( n_steps ) ] for x in range( batch_size ) ]
seq_len = np.asarray( [ random.randint( 1, n_steps - 1 ) for x in range( batch_size ) ] )

#print data
print seq_len

# input
x = tf.placeholder( "float", [ None, n_steps, n_input ] )
y = tf.placeholder( "float", [ None, n_classes ] )
early_stop = tf.placeholder( tf.int32, [ None ] )

#word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)

lstm_fw_cell = rnn_cell.BasicLSTMCell( lstm_size, forget_bias = 1.0 )
stacked_lstm_fw = rnn_cell.MultiRNNCell( [ lstm_fw_cell ] * number_of_layers )
initial_state_fw = state_fw = stacked_lstm_fw.zero_state( batch_size, tf.float32 )

lstm_bw_cell = rnn_cell.BasicLSTMCell( lstm_size, forget_bias = 1.0 )
stacked_lstm_bw = rnn_cell.MultiRNNCell( [ lstm_bw_cell ] * number_of_layers )
initial_state_bw = state_bw = stacked_lstm_bw.zero_state( batch_size, tf.float32 )

#istate_fw = tf.placeholder( "float", [ None, 2 * n_hidden ] )
#istate_bw = tf.placeholder( "float", [ None, 2 * n_hidden ] )
biases = {
    'hidden': tf.Variable(tf.random_normal( [ 2 * n_hidden ] ) ),
    'out': tf.Variable( tf.random_normal( [ n_classes ] ) )
}

weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2*n_hidden])),
    'out': tf.Variable(tf.random_normal( [ 2 * 2 * n_hidden, n_classes ] ) )
}

_x = tf.transpose( x, [ 1, 0, 2 ] )
#_x = x
_x = tf.reshape( _x, [ -1, n_input ] )
_x = tf.matmul( _x, weights['hidden']) + biases['hidden']
_x = tf.split( 0, n_steps, _x )

#print _x

outputs = rnn.bidirectional_rnn(
        stacked_lstm_fw,
        stacked_lstm_bw,
        _x,
        initial_state_fw = initial_state_fw,
        initial_state_bw = initial_state_bw,
        sequence_length = early_stop
        )

states = outputs[ -1 ]

pred = tf.matmul( outputs[ -1 ], weights[ 'out' ] ) + biases[ 'out' ]

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( pred, y ) ) # Softmax loss
optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate ).minimize( cost ) # Adam Optimizer

correct_pred = tf.equal( tf.argmax( pred, 1 ), tf.argmax( y , 1 ) )
accuracy = tf.reduce_mean( tf.cast( correct_pred, tf.float32 ) )


init = tf.initialize_all_variables()
training_iters = 1000

with tf.Session() as sess:
    sess.run( init )
    step = 1
    instances = 100

    while step * instances < training_iters:
        batch_xs = np.asarray( data )
        batch_ys = np.asarray( [ [ 1., 0. ] for _ in range( batch_size ) ] )

        print batch_ys
        print "*" * 20

        for instance in batch_xs:
            print instance
        print "*" * 20
        print seq_len

        epoch_dict = {
                    x : batch_xs,
                    y : batch_ys,
                    early_stop : seq_len,
                    #istate_fw : stacked_lstm_fw.zero_state( batch_size, tf.float32 ),
                    #istate_bw : stacked_lstm_bw.zero_state( batch_size, tf.float32 )
                    }

        sess.run(
                optimizer,
                feed_dict = epoch_dict
                )

        acc = sess.run(
                accuracy,
                feed_dict = epoch_dict
                )
        # Calculate batch loss
        loss = sess.run(
                cost,
                feed_dict = epoch_dict
                )
        predictions = sess.run(
                pred,
                feed_dict = epoch_dict
                )
        print "predictions"
        print predictions
        outs = sess.run(
                states,
                feed_dict = epoch_dict
                )
        print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
              ", Training Accuracy= " + "{:.5f}".format(acc)
        print "outs"
        for out in outs:
            print "*" * 100
            print out
        step += 1







