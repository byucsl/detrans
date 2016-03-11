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
from keras.models import Sequential, Graph
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
    return np.array( [ seq_to_indices( x, index ) for x in seqs ] )


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


# Take the totality of all the data and then split it into train, test, and
# validation datasets. All returned as numpy arrays.
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
    test_y = cds_seqs[ train_nb + test_nb : ]

    errw( "Done!\n" )

    return  train_x, train_y, test_x, test_y, validate_x, validate_y


# Build the deep BLSTM
# we're going to do this using a graph structure instead of sequential
# because it's easier to think about and do, sequential is weird :(
def build_model( nb_layers, nb_embedding_nodes, nb_lstm_nodes, aa_vocab_size, cds_vocab_size, maxlen, forwards_only ):
    errw( "Building model" )
    
    model = Graph()

    errw( "\tAdding initial input layer..." )
    model.add_input(
            name = "input",
            input_shape = ( maxlen, ),
            dtype = int
            )
    errw( "Done!\n" )

    errw( "\tAdding embedding layer..." )
    model.add_node( 
            Embedding( 
                aa_vocab_size,
                nb_embedding_nodes,
                mask_zero = True
                ), 
            name = "embedding",
            input = "input"
            )
    errw( "Done!\n" )

    # for each lstm layer add a layer to the model
    prev_forwards_input = "embedding"
    prev_backwards_input = "embedding"
    for i in range( nb_layers ):

        # set up the correct input names for each layer
        # if we're on the first iteration, the embedding layer is the
        # input layer
        # if we're past the first iteration it is previous lstm layer
        # that is the input layer
        if i > 0:
            prev_forwards_input = "forwards" + str( i - 1 )
            prev_backwards_input = "backwards" + str( i - 1 )

        errw( "\tCreating LSTM layer " + str( i + 1 ) + "\n" )
        errw( "\t\tAdding forwards layer built..." )

        model.add_node(
                LSTM(
                    nb_lstm_nodes,
                    return_sequences = True
                    ),
                name = "forwards" + str( i ),
                input = prev_forwards_input
                )
        errw( "Done!\n" )
        
        # if we're not doing a bidirectional lstm
        if not forwards_only:
            errw( "\t\tAdding backwards layer..." )
            model.add_node(
                    LSTM(
                        nb_lstm_nodes,
                        return_sequences = True
                        ),
                    name = "backwards" + str( i ),
                    input = prev_backwards_input
                    )
            errw( "Done!\n" )

    last_inputs = [ "forwards" + str( i ) ]
    if not forwards_only:
        last_inputs.append( "backwards" + str( i ) )

    errw( "\tAdding dense layer on top of LSTM layers..." )
    model.add_node(
            TimeDistributedDense(
                cds_vocab_size
                ),
            name = "timedistributeddense",
            inputs = last_inputs
            )
    errw( "Done!\n" )

    errw( "\tAdding softmax layer..." )
    model.add_node(
            Activation(
                "softmax"
                ),
            name = "activation",
            input = "timedistributeddense"
            )
    errw( "Done!\n" )

    errw( "\tAdding final output node to graph..." )
    model.add_output(
            name = "output",
            input = "activation"
            )
    errw( "Done!\n" )

    errw( "\tCompile constructed model..." )
    #model.compile( loss = 'categorical_crossentropy', optimizer = 'rmsprop' )
    model.compile(
            loss = { "output" : "categorical_crossentropy" },
            optimizer = "rmsprop"
            )
    errw( "Done!\n" )

    return model


# Take the constructed model and train it using the data provided
def train( model, train_x, train_y, validate_x, validate_y, nb_epochs, verbosity, model_save_prefix ):
    # train the model
    errw("Training...\n")

    for i in range( nb_epochs ):
        model.fit(
                {
                    "input" : train_x,
                    "output" : train_y,
                },
                nb_epoch = nb_epochs,
                batch_size = 1,
                verbose = verbosity,
                shuffle = False,
                validation_data = {
                    "input" : validate_x,
                    "output" : validate_y
                    }
                )

        # save the model and its weights
        errw( "\tSaving model..." )
        json_string = model.to_json()
        open( model_save_prefix + ".json", 'w' ).write( json_string )
        model.save_weights( model_save_prefix + ".h5", overwrite = True )
        errw( "Done!\n" )


# TODO: implement
# Take a trained model and print out the accuracy on the test set
def test_model( model, test_x, test_y ):
    # TODO: figure out how to show accuracy
    # this method doesn't seem to exist for graph models, it does for
    # sequential models though... which is odd
    pass


# TODO: implement
# Classify a given file, output is written to a file with the same name
# as the file_path + '.results' extension added. The output will be in
# fasta format
def classify( model, file_path ):
    pass


# Shorthand way to print to standard error
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
    errw( "\tModel training verbosity level: " + str( args.verbosity ) + "\n" )
    
    if args.classify:
        errw( "\tClassifying: " + args.clasify + "\n" )
    
    if args.forwards_only:
        errw( "\tForwards only network (not bidirectional)\n" )

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
    model = build_model( args.hidden_layers, args.embedding_nodes, args.lstm_nodes, aa_vocab_size, cds_vocab_size, max_seq_len, args.forwards_only )

    # train the model
    train( model, train_x, train_y, validate_x, validate_y, args.epochs, args.verbosity, args.model_save_path )

    # run model on test dataset and print accuracy
    test_model( model, test_x, test_y )

    # check if we need to classify another external file
    # classify if we need to and output the results to a file
    if args.classify:
        classify( model, args.classify )

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
            type = str,
            help = "A comma separated list of 3 values that denote how to split the input data between training, validation, and testing in that respective order (Default 70,15,15)."
            )
    parser.add_argument( '--forwards_only',
            type = bool,
            help = "Use only forwards LSTMs (Default False)."
            )
    parser.add_argument( '--verbosity',
            type = int,
            help = "Verbosity level when training the network (Default 1)."
            )
    default_model_save_path = "detrans_model." + time.strftime( "%Y-%m-%d" ) + "-" + time.strftime( "%H-%M-%S" )
    parser.set_defaults(
            hidden_layers = 1,
            embedding_nodes = 128,
            lstm_nodes = 128,
            model_save_path = default_model_save_path,
            epochs = 5,
            training_split = "70,15,15",
            forwards_only = False,
            verbosity = 1
            )

    args = parser.parse_args()

    # parameter checking
    split_total = sum( map( float, args.training_split.split( ',' ) ) )
    if split_total > 100.1 or split_total < 99.9:
        sys.exit( "ERROR: Dataset split " + args.training_split + " does not sum to 100!\n" )

    main( args )


