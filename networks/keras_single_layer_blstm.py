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

import sys, argparse, time, re
import numpy as np
from keras.models import Sequential
from keras.layers.core import Merge, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.layers.embeddings import Embedding

version = "0.1a"

def read_raw_text( file_path ):
    return open( file_path, 'r' ).read().strip()


def unique_words( text ):
    #return set( re.findall( r"[\w+']+", text ) )
    return set( text.replace( '\n', ' ' ).split( ' ' ) )


# This also includes a "":0 entry in the dictionary
def word_to_number( word_set ):
    index = dict( ( c, i + 1 ) for i, c in enumerate( word_set ) )
    index[ "" ] = 0
    return index


def number_to_word( index ):
    return { v : k for k, v in codon_index.items() }


def parse_seqs( raw_text ):
    return [ x.strip() for x in raw_text.split( '\n' ) ]


def seq_to_indices( seq, index ):
    return [ index[ x ] for x in seq.split( ' ' ) ]


def seqs_to_indices( seqs, index ):
    return [ seq_to_indices( x, index ) for x in seqs ]


def seqs_to_one_hot( maxlen, seqs, index ):
    encodings = np.zeros( ( len( seqs ), maxlen, len( index ) ), np.bool )
    for sidx, seq in enumerate( seqs ):
        seq = seq.split()
        offset = maxlen - len( seq )

        for cidx, codon in enumerate( seq ):
            encodings[ sidx ][ offset + cidx ][ index[ codon ] ] = 1
    return encodings


def load_data( amino_acid_path, codons_path ):
    sys.stderr.write( "Loading data\n" )
    
    sys.stderr.write( "\tLoading amino acid file..." )
    aa_raw_text = read_raw_text( amino_acid_path )
    amino_acids_vocab = unique_words( aa_raw_text )
    aa_index = word_to_number( amino_acids_vocab )
    aa_seqs = parse_seqs( aa_raw_text )
    sys.stderr.write( "Done!\n" )

    sys.stderr.write( "\tLoading codon file..." )
    cds_raw_text = read_raw_text( codons_path )
    codons_vocab = unique_words( cds_raw_text )
    cds_index = word_to_number( codons_vocab )
    cds_seqs = parse_seqs( cds_raw_text )
    sys.stderr.write( "Done!\n" )

    return aa_index, aa_seqs, cds_index, cds_seqs


def split_train_test_validate( split_vals ):
    pass


def build_model():
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

    return model


def train( model ):
    # train the model
    print("Train...")

    for i in range( nb_epoches ):
        model.fit( train_x, train_y, batch_size=1, nb_epoch=1, validation_data=(train_x, train_y), verbose=1, show_accuracy=True)


def errw( string ):
    sys.stderr.write( string )


def main( args ):
    errw( "Detrans " + version + "\n" )
    errw( "Amino acid file: " + args.amino_acids_path + "\n" )
    errw( "Codons file: " + args.codons_path + "\n" )
    errw( "Number epochs for training: " + str( args.epochs ) + "\n" )
    errw( "Hidden layers: " + str( args.hidden_layers ) + "\n" )
    errw( "Hidden nodes: " + str( args.hidden_nodes ) + "\n" )
    errw( "Model save path: " + args.model_save_path + "\n" )
    if args.classify:
        errw( "Classifying: " + args.clasify + "\n" )

    # load data
    aa_index, aa_seqs, cds_index, cds_seqs = load_data( args.amino_acids_path, args.codons_path )

    # prepare needed parameters for model training
    aa_vocab_size = len( aa_index )
    max_seq_len = ( len( max( aa_seqs, key = len ) ) + 1 ) / 2

    sys.stderr.write( "Amino acid vocab size: " + str( aa_vocab_size ) + "\n" )
    sys.stderr.write( "Maximum sequence length: " + str( max_seq_len ) + "\n" )

    # prepare text for model training
    aa_seqs = seqs_to_indices( aa_seqs, aa_index )
    aa_seqs = pad_sequences( aa_seqs, maxlen = max_seq_len )
    cds_seqs = seqs_to_one_hot( max_seq_len, cds_seqs, cds_index )

    errw( "Done!\n" )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Build and train a model for amino acid detranslation."
            )
    parser.add_argument( 'amino_acids_path',
            type = str,
            help = "Path to the file containing the properly formatted amino acid sequences."
            )
    parser.add_argument( 'codons_path',
            type = str,
            help = "Path to the file containing the properly formatted codon sequences."
            )
    parser.add_argument( '--classify',
            type = str,
            help = "Path to a file to detranslate that is not training, validation, nor test. Results will be written to the given path with a .results extension added."
            )
    parser.add_argument( '--epochs',
            help = "The number of epochs to train for (Default 5)."
            )
    parser.add_argument( '--model_save_path',
            type = str,
            help = "Path to save the trained model (Default YYYY-MM-DD-HH-MM-SS)."
            )
    parser.add_argument( '--hidden_layers',
            help = "Number of hidden layers to use (Default 1)."
            )
    parser.add_argument( '--hidden_nodes',
            help = "Number of hidden nodes in each layer (Default 128)."
            )
    parser.add_argument( '--training_split',
            help = "A comma separated list of 3 values that denote how to split the input data between training, validation, and testing in that respective order (Default 70,15,15)."
            )
    default_model_save_path = time.strftime( "%Y-%m-%d" ) + "-" + time.strftime( "%H-%M-%S" )
    parser.set_defaults(
            hidden_layers = 1,
            hidden_nodes = 128,
            model_save_path = default_model_save_path,
            epochs = 5
            )

    args = parser.parse_args()

    main( args )


