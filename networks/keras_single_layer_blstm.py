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
from keras.models import Sequential
from keras.layers.core import Merge, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

# set model params
nb_epoches = 10

# parse the incoming data
aas = []
sys.stderr.write( "Reading data\n" )
sys.stderr.write( "\tReading amino acid seqs file..." )
raw_text = open( sys.argv[ 1 ] ).read()
aas_set = set( raw_text.split() )
print aas_set
for line in raw_text:
    aas.append( line.strip() )
sys.stderr.write( " Done! (" + str( len( aas ) ) + " amino acid seqs read)\n" )

cds = []
sys.stderr.write( "\tReading codon seqs file..." )
raw_text = open( sys.argv[ 2 ] ).read()
cds_set = set( raw_text.split() )
for line in raw_text:
    cds.append( line.strip() )
sys.stderr.write( " Done! (" + str( len( cds ) ) + " codon seqs read)]n" )

# create the training sets, validation sets, testing sets
sys.stderr.write( "Creating train, validation and test sets\n" )

sys.stderr.write( "\tCreating train..." )
train_x = []
train_y = []

train_x = aas[ : 10 ]
train_y = cds[ : 10 ]

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
left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_shape=(99, 13)))
print( "Left built..." )

right = Sequential()
right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_shape=(99, 13), go_backwards=True))
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
    print "Epoch:", str( i )
    model.fit( train_x, train_y, batch_size=1, nb_epoch=nb_epoches, validation_data=(train_x, train_y), verbose=1, show_accuracy=True)
