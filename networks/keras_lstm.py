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
from keras.layers.core import Merge, Activation, TimeDistributedDense, Dropout, Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.layers.embeddings import Embedding


# set model params
epoches = int( sys.argv[ 3 ] )
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

print "number of amino acids:\t", str( len_aas )
print aas_set

#for key, val in aa_indices.iteritems():
#    print key, val

for line in raw_text.split( '\n' ):
    aas.append( line.strip() )
    if len( aas[ -1 ].split() ) > maxlen:
        maxlen = len( aas[ -1 ].split() )

sys.stderr.write( " Done! (" + str( len( aas ) ) + " amino acid seqs read)\n" )

#aas_p = np.zeros( ( len( aas ), maxlen, len_aas ), dtype = np.bool )
aas_p = []
for s_idx, seq in enumerate( aas ):
    #print "seq:\t", seq.split()
    #seq_one_hot = one_hot( seq, len_aas, lower = False, split = ' ' )
    seq_one_hot = []
    for acid in seq.split():
        seq_one_hot.append( aa_indices[ acid ] )
    aas_p.append( seq_one_hot )
    #print len( seq.split() )
    #print len( seq_one_hot )
    print zip( seq.split(), seq_one_hot )
    #for r_idx, res in enumerate( seq_one_hot ):
    #    aas_p[ s_idx, r_idx, res ] = 1
    #aas_p.append( text_to_word_sequence( seq, lower = False, split = " " ) )
    print aas_p[ -1 ]
aas_p = pad_sequences( aas_p, maxlen = maxlen )
print "*" * 10
print aas_p

cds = []
sys.stderr.write( "\tReading codon seqs file..." )
raw_text = open( sys.argv[ 2 ] ).read().strip()
cds_set = set( raw_text.split() )
len_codons = len( cds_set )
codon_indices = dict( ( c, i ) for i, c in enumerate( cds_set ) )
idx_to_codon = {v: k for k, v in codon_indices.items()}

#for key, val in codon_indices.iteritems():
#    print key, val

for line in raw_text.split( '\n' ):
    cds.append( line.strip() )
sys.stderr.write( " Done! (" + str( len( cds ) ) + " codon seqs read)]\n" )

sys.stderr.write( "Maxlen:\t" + str( maxlen ) + "\n" )

print cds
cds_p = []
cds_p = np.zeros( ( len( aas_p ), maxlen, len_codons ), np.bool )
for sidx, seq in enumerate( cds ):
    #print "*" * 20
    #cds_p.append( text_to_word_sequence( seq, lower = False, split = ' ' ) )
    #cds_p.append( one_hot( seq, len_codons + 1, lower = False, split = ' ' ) )
    seq_one_hot = []
    offset = maxlen - len( seq.split() )
    for cidx, codon in enumerate( seq.split() ):
        #seq_one_hot.append( codon_indices[ codon ] )
        cds_p[ sidx ][ offset + cidx ][ codon_indices[ codon ] ] = 1
    #cds_p.append( seq_one_hot )
    #print zip( cds_t, cds_1 )
#cds_p = pad_sequences( cds_p, maxlen = maxlen )
#cds_p = np.zeros( ( len( aas_p ), maxlen, len_codons ), np.bool )
print cds_p

for i in range( len( cds_p ) ):
    print "*" * 20
    print len( cds_p[ i ] )
    print len( aas_p[ i ] )

# pad the inputs for rnn usage
#aas = pad_sequences( aas, maxlen = maxlen )
#cds_p = pad_sequences( cds_p, maxlen = maxlen )
#print "*" * 20
#print cds_p

# create the training sets, validation sets, testing sets
sys.stderr.write( "Creating train, validation and test sets\n" )

sys.stderr.write( "\tCreating train..." )
#train_x = np.zeros((len(aas), maxlen, len(chars)), dtype=np.bool)
#train_y = X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
train_x = aas_p
train_y = cds_p
hidden_units = 10
nb_classes = 10


print('Build model...')
model = Sequential()
model.add(Embedding(len_aas, 128))
model.add(LSTM(len_codons, return_sequences=True))
model.add(TimeDistributedDense(len_codons))
model.add(Activation('softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

for iteration in range(1, epoches):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit( train_x, train_y, batch_size=1, nb_epoch=1)

results = model.predict( train_x )

for result in results:
    print( "*" * 20 )
    generated_seq = ""
    for pos in result:
        codon_idx = np.argmax( pos )
        codon_prob = np.amax( pos )
        codon = idx_to_codon[ codon_idx ]
        print( codon + "\t" + str( codon_idx ) + "\t" + str( codon_prob ) )
        generated_seq += codon
    print( generated_seq )



