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
    sys.stderr.write( "\nLoading data\n" )
    
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


def split_train_test_validate( split_vals, aa_seqs, cds_seqs ):
    errw( "\nSplitting Data into train, test, validate (" + split_vals +")\n" )
    errw( "\tSplitting..." )
    split_vals = map( float, split_vals.split( ',' ) )
    split_vals = [ x /  sum( split_vals ) for x in split_vals ]

    total_nb = len( aa_seqs )

    # num seqs to get for train, test, validate
    # convert these to ints so that we don't have floats for indices
    # TODO: check to see what happens if indices don't divide evenlty
    train_nb = int( split_vals[ 0 ] * total_nb )
    test_nb = int( split_vals[ 1 ] * total_nb )
    validate_nb = int( split_vals[ 2 ] * total_nb )

    # partition
    train_x = aa_seqs[ : train_nb ]
    train_y = cds_seqs[ : train_nb ]
    validate_x = aa_seqs[ train_nb : train_nb + test_nb ]
    validate_y = cds_seqs[ train_nb : train_nb + test_nb ]
    test_x = aa_seqs[ train_nb + test_nb : ]
    test_y = aa_seqs[ train_nb + test_nb : ]

    errw( "Done!\n" )

    return  train_x, train_y, test_x, test_y, validate_x, validate_y


def fork( model, nb_layers ):
    fork = []
    for i in range( nb_layers ):
        f = Sequential()
        f.add( model )
        forks.append( f )
    return forks


def build_model( nb_layers, nb_embedding_nodes, nb_lstm_nodes, aa_vocab_size, cds_vocab_size, maxlen ):
    errw( "Building model" )
    
    # create the forwards and backwards networks
    # add lstm layers in for loop
   
    errw( "\tBuiling embedding layer..." )
    # forwards embedding layer
    forwards = Sequential()
    forwards.add( Embedding( aa_vocab_size, nb_embedding_nodes, masking_zero = True ) )

    # backwards embedding layer
    backwards = Sequential()
    backwards.add( Embedding( max_features, hidden_units, input_length = maxlen ) )
    errw( "Done!\n" )

    # for each lstm layer add a layer to the model
    for i in range( nb_layers ):
        errw( "\tCreating layer " + str( i + 1 ) + "\n" )
        errw( "\t\tForwards network built..." )
        forwards.add( LSTM( nb_lstm_nodes, return_sequences = True ) )
        #left.add( LSTM( output_dim = hidden_units, init = 'uniform', inner_init = 'uniform',
        #               forget_bias_init = 'one', return_sequences = True, activation = 'tanh',
        #               inner_activation = 'sigmoid', input_shape = ( maxlen, len_aas ) ) )
        errw( "Done!\n" )
        
        errw( "\t\tBackwards network built..." )
        backwards.add( LSTM( nb_lstm_nodes, return_sequences = True, go_backwards = True ) )
        #right.add( LSTM( output_dim = hidden_units, init = 'uniform', inner_init = 'uniform',
        #               forget_bias_init = 'one', return_sequences = True, activation = 'tanh',
        #               inner_activation = 'sigmoid', input_shape = ( maxlen, len_aas ),
        #               go_backwards = True ) )
        errw( "Done!\n" )

    errw( "\tMerging forwards and backwards..." )
    model = Sequential()
    model.add( Merge( [ forwards, backwards ], mode = 'sum' ) )
    errw( "Done!\n" )

    #model.add(TimeDistributedDense(nb_classes))
    #print( "Dense added..." )

    #model.add(Activation('softmax'))
    #print( "Activation added..." )

    #sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
    #print( "Loss function defined..." )

    #model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #print( "Model compiled..." )

    return model


def train( model ):
    # train the model
    print("Train...")

    for i in range( nb_epoches ):
        model.fit( train_x, train_y, batch_size=1, nb_epoch=1, validation_data=(train_x, train_y), verbose=1, show_accuracy=True)


def errw( string ):
    sys.stderr.write( string )


def main( args ):
    errw( "Detrans " + version + "\n\n" )
    errw( "Input parameters:\n" )
    errw( "\tAmino acid file: " + args.amino_acids_path + "\n" )
    errw( "\tCodons file: " + args.codons_path + "\n" )
    errw( "\tNumber epochs for training: " + str( args.epochs ) + "\n" )
    errw( "\tHidden layers: " + str( args.hidden_layers ) + "\n" )
    errw( "\tEmbedding layer nodes: " + str( args.embedding_nodes ) + "\n" )
    errw( "\tLSTM output nodes: " + str( args.lstm_nodes ) + "\n" )
    errw( "\tModel save path prefix: " + args.model_save_path + "\n" )
    errw( "\tData splits: " + args.training_split + "\n" )
    if args.classify:
        errw( "\tClassifying: " + args.clasify + "\n" )

    # load data
    aa_index, aa_seqs, cds_index, cds_seqs = load_data( args.amino_acids_path, args.codons_path )

    # prepare needed parameters for model training
    aa_vocab_size = len( aa_index )
    cds_vocab_size = len( cds_index )
    max_seq_len = ( len( max( aa_seqs, key = len ) ) + 1 ) / 2

    errw( "Amino acid vocab size: " + str( aa_vocab_size ) + "\n" )
    errw( "Codons vocab size: " + str( cds_vocab_size ) + "\n" )
    errw( "Maximum sequence length: " + str( max_seq_len ) + "\n" )

    # prepare text for model training
    errw( "Prepare sequences for input into learning algorithms\n" )
    errw( "\tConvert amino acid sequences..." )
    aa_seqs = seqs_to_indices( aa_seqs, aa_index )
    errw( "Done!\n" )
    errw( "\tPad amino acid sequences..." )
    aa_seqs = pad_sequences( aa_seqs, maxlen = max_seq_len )
    errw( "Done!\n" )
    errw( "\tOne-hot encode codon sequences..." )
    cds_seqs = seqs_to_one_hot( max_seq_len, cds_seqs, cds_index )
    errw( "Done!\n" )

    # split the data into train, validation, and test data
    train_x, train_y, test_x, test_y, validate_x, validate_y = split_train_test_validate( args.training_split, aa_seqs, cds_seqs, )

    errw( "Total instances: " + str( len( aa_seqs ) ) + "\n" )
    errw( "\tTrain instances:    " + str( len( train_x ) ) + "\n" )
    errw( "\tTest instances:     " + str( len( test_x ) )  + "\n" )
    errw( "\tValidate instances: " + str( len( validate_x ) ) + "\n" )

    # build model
    model = build_model( args.hidden_layers, args.embedding_nodes, args.lstm_nodes, aa_vocab_size, cds_vocab_size, max_seq_len )

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
            type = int,
            help = "The number of epochs to train for (Default 5)."
            )
    parser.add_argument( '--model_save_path',
            type = str,
            help = "Path to save the trained model (Default YYYY-MM-DD-HH-MM-SS)."
            )
    parser.add_argument( '--hidden_layers',
            type = int,
            help = "Number of hidden layers (LSTMs) to use (Default 1)."
            )
    parser.add_argument( '--lstm_nodes',
            type = int,
            help = "Number of output nodes to use in LSTM layers of network (Default 128)."
            )
    parser.add_argument( '--embedding_nodes',
            type = int,
            help = "Number of hidden nodes in each layer (Default 128)."
            )
    parser.add_argument( '--training_split',
            help = "A comma separated list of 3 values that denote how to split the input data between training, validation, and testing in that respective order (Default 70,15,15)."
            )
    default_model_save_path = "detrans_model." + time.strftime( "%Y-%m-%d" ) + "-" + time.strftime( "%H-%M-%S" )
    parser.set_defaults(
            hidden_layers = 1,
            embedding_nodes = 128,
            lstm_nodes = 128,
            model_save_path = default_model_save_path,
            epochs = 5,
            training_split = "70,15,15"
            )

    args = parser.parse_args()

    # parameter checking
    split_total = sum( map( float, args.training_split.split( ',' ) ) )
    if split_total > 100.1 or split_total < 99.9:
        sys.exit( "ERROR: Dataset split " + args.training_split + " does not sum to 100!\n" )

    main( args )


