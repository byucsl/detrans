'''
This is a single layer bidirectional lSTM.
It was taken from example code found here: https://github.com/fchollet/keras/issues/1629
Modified to test on the amino acid detranslation problem

PARAMETERS:
    - amino acid seqs
    - codons seqs

modified by stanley fujimoto (masakistan)
4 mar 2016
'''


import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Merge, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.layers.embeddings import Embedding


# set model params
nb_epoches = 10
maxlen = 0
max_features = 100

# parse the incoming data
aas = []
sys.stderr.write( "Reading data\n" )
sys.stderr.write( "\tReading amino acid seqs file..." )
raw_text = open( sys.argv[ 1 ] ).read().strip()
aas_set = set( raw_text.split() )
len_aas = len( aas_set )
aa_indices = dict( ( c, i ) for i, c in enumerate( aas_set ) )

#for key, val in aa_indices.iteritems():
#    print key, val

for line in raw_text.split( '\n' ):
    aas.append( line.strip() )
    if len( aas[ -1 ].split() ) > maxlen:
        maxlen = len( aas[ -1 ].split() )

sys.stderr.write( " Done! (" + str( len( aas ) ) + " amino acid seqs read)\n" )

aas_p = []
for seq in aas:
    aas_p.append( one_hot( seq, len_aas + 1, lower = False, split = ' ' ) )
aas_p = pad_sequences( aas_p, maxlen = maxlen )

cds = []
sys.stderr.write( "\tReading codon seqs file..." )
raw_text = open( sys.argv[ 2 ] ).read().strip()
cds_set = set( raw_text.split() )
len_codons = len( cds_set )
codon_indices = dict( ( c, i ) for i, c in enumerate( cds_set ) )

#for key, val in codon_indices.iteritems():
#    print key, val

for line in raw_text.split( '\n' ):
    cds.append( line.strip() )
sys.stderr.write( " Done! (" + str( len( cds ) ) + " codon seqs read)]\n" )

sys.stderr.write( "Maxlen:\t" + str( maxlen ) )

print cds
cds_p = []
for seq in cds:
    print "*" * 20
    #cds_p.append( text_to_word_sequence( seq, lower = False, split = ' ' ) )
    cds_p.append( one_hot( seq, len_codons + 1, lower = False, split = ' ' ) )
    #print zip( cds_t, cds_1 )

# pad the inputs for rnn usage
#aas = pad_sequences( aas, maxlen = maxlen )
cds_p = pad_sequences( cds_p, maxlen = maxlen )
print "*" * 20
print cds_p

# create the training sets, validation sets, testing sets
sys.stderr.write( "Creating train, validation and test sets\n" )

sys.stderr.write( "\tCreating train..." )
#train_x = np.zeros((len(aas), maxlen, len(chars)), dtype=np.bool)
#train_y = X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
train_x = aas_p
train_y = cds_p
'''
for aas_idx, seq in enumerate( aas ):
    train_x.append( np.zeros( ( len( seq ), len_aas ), dtype = np.bool ) )
    train_y.append( np.zeros( ( len( seq ), len_codons ), dtype = np.bool ) )
    for aa_idx, cur_aa in enumerate( seq ):
        cur_codon = cds[ aas_idx ][ aa_idx ]
        train_x[ -1 ][ aa_idx ][ aa_indices[ cur_aa ] ] = 1
        train_y[ -1 ][ aa_idx ][ codon_indices[ cur_codon ] ] = 1

train_x = np.array( train_x[ : 1 ] )
train_y = np.array( train_y[ : 1 ] )'''

#print "*" * 20
#for i in np.ndenumerate( train_x[ 0 ][ 0 ] ):
#    print i
#
#print "*" * 20
#for i in np.ndenumerate( train_y[ 0 ][ 0 ] ):
#    print i
#
#print "*" * 20


sys.stderr.write( " Done!\n" )

sys.stderr.write( "\tCreating validation..." )
validate_x = []
validate_y = []
sys.stderr.write( " Done!\n" )

sys.stderr.write( "\tCreating test..." )
test_x = []
test_y = []
sys.stderr.write( " Done!\n" )


# set dimensions of network
hidden_units = 10
nb_classes = 10


left = Sequential()
left.add( Embedding( max_features, hidden_units, input_length = maxlen ) )
left.add( LSTM( output_dim = hidden_units, init = 'uniform', inner_init = 'uniform',
               forget_bias_init = 'one', return_sequences = True, activation = 'tanh',
               inner_activation = 'sigmoid', input_shape = ( maxlen, len_aas ) ) )
print( "Left built..." )

right = Sequential()
right.add( Embedding( max_features, hidden_units, input_length = maxlen ) )
right.add( LSTM( output_dim = hidden_units, init = 'uniform', inner_init = 'uniform',
               forget_bias_init = 'one', return_sequences = True, activation = 'tanh',
               inner_activation = 'sigmoid', input_shape = ( maxlen, len_aas ), go_backwards = True ) )
print( "Right built..." )

model = Sequential()
model.add(Merge([left, right], mode='sum'))
print( "Merged..." )

model.add(TimeDistributedDense(nb_classes))
print( "Dense added..." )

model.add(Activation('softmax'))
print( "Activation added..." )

sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
print( "Loss function defined..." )

model.compile(loss='categorical_crossentropy', optimizer=sgd)
print( "Model compiled..." )

# train the model
print("Train...")

for i in range( nb_epoches ):
    model.fit( train_x, train_y, batch_size=1, nb_epoch=1, validation_data=(train_x, train_y), verbose=1, show_accuracy=True)


