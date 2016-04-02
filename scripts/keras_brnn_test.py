import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Graph
from keras.layers.core import Dense, Lambda
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
import numpy as np


def reverse_func(x):
    import keras.backend as K
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
    return K.permute_dimensions(rev, (1, 0, 2))

reverse = Lambda(reverse_func) # Add this layer after your go backwards LSTM

epochs = 10
maxlen = 6

#a = np.array( [ [ 0 ] ] * ( maxlen / 2 ) )
#a.fill( 2 )

a = np.array( [ [ 1 ], [ 2 ], [ 3 ] ]  )
#print a

x = np.array( [ a ] )
x = pad_sequences( x, maxlen = maxlen )

print x
print x.shape

g = Graph()
g.add_input( name = 'input_f', input_shape = ( maxlen, 1,  ) )
g.add_input( name = 'input_b', input_shape = ( maxlen, 1,  ) )
g.add_node(
        SimpleRNN(
            1,
            return_sequences = True,
            inner_init = 'identity',
            init = 'identity',
            activation = 'linear' ),
        name = 'forwards',
        input = 'input_f'
        )
g.add_node( 
        SimpleRNN( 
            1,
            return_sequences = True,
            inner_init = 'identity',
            init = 'identity',
            activation = 'linear',
            go_backwards = True
            ),
        name = 'backwards_rnn',
        input = 'input_b'
        )
g.add_node(
    Lambda(
        reverse_func
        ),
    name = 'backwards',
    input = 'backwards_rnn'
    )
g.add_output( name = 'output', inputs = [ 'forwards', 'backwards' ] )
#g.add_output( name = 'output', input = 'backwards' )

for name, node in g.nodes.iteritems():
    print name
    #print node.get_weights()
    #print len( node.get_weights() )

    w_1 = []

    for weight in node.get_weights():
        #print "\t" + str( weight.shape )
        t_a = np.ones_like( weight )
        w_1.append( t_a )
    #print node.get_weights()
    #print w_1
    #node.set_weights( w_1 )
    print node.get_weights()

g.compile(optimizer='rmsprop', loss={'output':'mse'})

p = g.predict( { 'input_f' : x, 'input_b' : x } )

print "*" * 20

print p[ 'output' ]

print "Done!"

