'''
This is a multi-layer, deep bidirectional lSTM.
It was inspired by teh example code found here: https://github.com/fchollet/keras/issues/1629 but has been heavily modified.
Modified to test on the amino acid detranslation problem

PARAMETERS:
    - amino acid seqs
    - codons seqs

modified by stanley fujimoto (masakistan)
4 mar 2016
'''

import sys, argparse, datetime, time, re, pickle, copy
import numpy as np
from sklearn.utils import shuffle
from keras.engine.topology import Layer
from keras.models import model_from_json, Model, Sequential
from keras.layers import Lambda, Input, merge, TimeDistributed, Dense, Embedding
from keras.layers import LSTM, GRU, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.utils.layer_utils import layer_from_config
import pandas as pd
from Bio.Seq import translate
from Bio.Seq import CodonTable

sys.setrecursionlimit(10000)

# Global variables
version = "0.1a"

all_codons = [
                'AAA', 'AAC', 'AAT', 'AAG',
                'ACA', 'ACC', 'ACT', 'ACG',
                'ATA', 'ATC', 'ATT', 'ATG',
                'AGA', 'AGC', 'AGT', 'AGG',
                'CAA', 'CAC', 'CAT', 'CAG',
                'CCA', 'CCC', 'CCT', 'CCG',
                'CTA', 'CTC', 'CTT', 'CTG',
                'CGA', 'CGC', 'CGT', 'CGG',
                'TAA', 'TAC', 'TAT', 'TAG',
                'TCA', 'TCC', 'TCT', 'TCG',
                'TTA', 'TTC', 'TTT', 'TTG',
                'TGA', 'TGC', 'TGT', 'TGG',
                'GAA', 'GAC', 'GAT', 'GAG',
                'GCA', 'GCC', 'GCT', 'GCG',
                'GTA', 'GTC', 'GTT', 'GTG',
                'GGA', 'GGC', 'GGT', 'GGG'
            ]

bacterial_codon_table = CodonTable.unambiguous_dna_by_id[ 11 ]

# this is the list of all possible amino acids
all_aas = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z',
        '*'
        ]

all_codons = list( sorted( all_codons ) )

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

class BrnnAlign( Layer ):
    #def __init__(self, func, *args, **kwargs):
    def __init__(self, *args, **kwargs):
        #self.func = func
        self.supports_masking = True
        super( BrnnAlign, self ).__init__( *args, **kwargs )

    def compute_mask( self, x, mask = None ):
        return mask[ :: -1 ]

    def call( self, x, mask = None ):
        return x[ :, :: -1, : ]


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
    return { v : k for k, v in index.items() }


def parse_seqs( raw_text, seq_len_cutoff ):
    return [ x.strip() for x in raw_text.split( '\n' ) if len( x.split( ' ' ) ) <= seq_len_cutoff or seq_len_cutoff == 0 ]


def seq_to_indices( seq, index ):
    return [ index[ x ] for x in seq.split( ' ' ) ]


def seqs_to_indices( seqs, index ):
    return np.array( [ seq_to_indices( x, index ) for x in seqs ] )


def seqs_to_one_hot( maxlen, seqs, index ):
    encodings = np.zeros( ( len( seqs ), maxlen, len( index ) ), np.int )
    for sidx, seq in enumerate( seqs ):
        seq = seq.split()
        #offset = maxlen - len( seq )

        for cidx, codon in enumerate( seq ):
            try:
                encodings[ sidx ][ cidx ][ index[ codon ] ] = 1
            except KeyError:
                # This should only happen if the pretrained network didn't have all the
                # codons that are being used in later training. This should be fixed
                # in current releases.
                encodings[ sidx ][ cidx ][ 0 ] = 1
    return encodings


def load_data( amino_acid_path, codons_path, seq_len_cutoff ):
    sys.stderr.write( "\nLoading data\n" )
    
    sys.stderr.write( "\tLoading amino acid file..." )
    aa_raw_text = read_raw_text( amino_acid_path )
    #amino_acids_vocab = unique_words( aa_raw_text )
    amino_acids_vocab = set( all_aas )
    aa_index = word_to_number( amino_acids_vocab )
    aa_seqs = parse_seqs( aa_raw_text, seq_len_cutoff )
    sys.stderr.write( "Done!\n" )

    sys.stderr.write( "\tLoading codon file..." )
    cds_raw_text = read_raw_text( codons_path )
    #codons_vocab = unique_words( cds_raw_text )
    codons_vocab = set( all_codons )
    cds_index = word_to_number( codons_vocab )
    cds_seqs = parse_seqs( cds_raw_text, seq_len_cutoff )
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


# This method will modify the model architecture json file so that it freezes the weights
# for all the deep layers and leaves the last layer, the classification layer, available
# for training
def freeze_network( model ):
    errw( "\tFreezing weights..." )

    # freeze all but the last layer
    for layer in model.layers[ : -1 ]:
        layer.trainable = False

    errw( "Done!\n" )


def reset_weights( model ):
    errw( "\tResetting layer weights for one-shot learning..." )

    output_shape = model.layers[ -1 ].output_shape[ -1 ]

    errw( "\t\tRemoving old classification layer..." )
    #model.layers.pop()
    #model.layers[ -1 ].outbound_nodes = []
    #model.outputs = [ model.layers[ -1 ].output ]
    #model.built = False
    errw( "Done!\n" )

    errw( "\t\tAdding new dense layer..." )
    #model.add(
    #        TimeDistributed(
    #            Dense(
    #                output_shape,
    #                activation = "softmax"
    #                )
    #            )
    #        )
    errw( "Done!\n" )

    errw( "\t\tRecompiling model..." )
    model.compile(
            loss = "categorical_crossentropy",
            optimizer = "adam",
            metrics = [ "categorical_accuracy" ],
            sample_weight_mode = 'temporal',
            )
    errw( "Done!\n" )


# load a model from disk
def load_model( model_path, one_shot ):
    errw( "Loading model with prefix: " + model_path + "\n" )

    errw( "\tLoading model architecture..." )
    # read the model architecture from the json file
    model_arch = open( model_path + ".json" ).read()
    model = model_from_json( model_arch )
    errw( "Done!\n" )

    if args.one_shot:
        # change parameters so that certain layers are not trained.
        freeze_network( model )

    errw( "\tLoading model weights..." )
    model.load_weights( model_path + ".h5" )
    errw( "Done!\n" )

    if args.one_shot:
        # reset the weights of the last layer
        reset_weights( model )

    errw( "\tLoading amino acid index..." )
    aa_index = pickle.load( open( model_path + ".aa_index.p", "rb" ) )
    errw( "Done!\n" )

    errw( "\tLoading codons index..." )
    cds_index = pickle.load( open( model_path + ".cds_index.p", "rb" ) )
    errw( "Done!\n" )

    return model, aa_index, cds_index


# Build the deep BLSTM
# we'll use the keras functional API because Graph has been deprecated and sequential
# is something...?
def build_model( nb_layers, nb_embedding_nodes, nb_lstm_nodes, aa_vocab_size, cds_vocab_size, maxlen, forwards_only, node_type, drop_w, drop_u ):
    errw( "Building model" )

    errw( "\tCreating sequential model..." )
    model = Sequential()
    errw( "Done!\n" )

    errw( "\tAdding embedding layer..." )
    model.add(
            Embedding(
                aa_vocab_size, nb_embedding_nodes, mask_zero = True )
            )
    errw( "Done!\n" )

    errw( "\tAdding " + str( nb_layers ) + " recurrent layers...\n" )
    for i in range( nb_layers ):
        errw( "\t\tAdding layer " + str( i + 1 ) + "..." )

        if i == 0:
            merge_mode = "concat"
        else:
            merge_mode = "mul"

        model.add(
                Bidirectional(
                    LSTM(
                        nb_lstm_nodes,
                        return_sequences = True,
                        dropout_W = drop_w,
                        dropout_U = drop_u
                        ),
                    merge_mode = merge_mode
                    )
                )
        errw( "Done!\n" )
    errw( "\tDone adding recurrent layers!\n" )

    errw( "\tAdding dense layer on top of the recurrent layers..." )
    model.add(
            TimeDistributed(
                Dense(
                    cds_vocab_size,
                    activation = "softmax"
                    )
                )
            )
    errw( "Done!\n" )

    errw( "\tCompile constructed model..." )
    model.compile(
            loss = "categorical_crossentropy",
            optimizer = "adam",
            metrics = [ "categorical_accuracy" ],
            sample_weight_mode = 'temporal',
            )
    errw( "Done!\n" )

    return model


# Take the constructed model and train it using the data provided
def train( model, train_x, train_y, validate_x, validate_y, nb_epochs, verbosity, model_save_prefix, idx_to_codon, no_save, save_confusion_matrix ):
    #print train_x[ 0 ][ 0 ]
    #print train_y[ 0 ][ 0 ]
    #print "train_x shape:", train_x.shape
    #print "train_y shape:", train_y.shape

    x_sample_weight = np.array( train_x )
    x_sample_weight[ x_sample_weight > 0 ] = 1

    # train the model
    errw("Training...\n")

    validate_y_labels = np.argmax( validate_y, axis = 1 )

    for i in range( nb_epochs ):
        if verbosity > 0:
            errw( "\tTraining epoch: " + str( i + 1 ) + "/" + str( nb_epochs ) + "\n" )

        # remove validation because categorical accuracy for masked inputs doesn't work
        # so, we'll do our own calculation withe calc_category_accuracy
        # it will look not only at codon accuracy but amino acid accuracy as well
        # this should save time as well because we're not calculating the the labels for
        # the validations set twice
        model.fit(
                train_x,
                train_y,
                nb_epoch = 1,
                verbose = verbosity,
                #validation_data = ( validate_x, validate_y ),
                batch_size = 32,
                sample_weight = x_sample_weight
                )

        predictions = model.predict( validate_x )

        aa_acc, codon_acc, aa_cf, codon_cf = calc_category_accuracy( predictions, validate_y, idx_to_codon, i, model_save_prefix )
        errw( "\n\t\taccuracy metrics:\n" )
        errw( "\t\t\taa acc: \t" + str( aa_acc ) + "\n" )
        errw( "\t\t\tcds acc:\t" + str( codon_acc ) + "\n" )

        # print confusion matrices to disk

        if save_confusion_matrix:
            aa_cf.to_csv( model_save_prefix + ".aa_cf." + str( i ) + ".csv" )
            codon_cf.to_csv( model_save_prefix + ".codon_cf." + str( i ) + ".csv" )

        # Save the model
        if not no_save:
            errw( "\t\tSaving model..." )
            json_string = model.to_json()
            open( model_save_prefix + ".json", 'w' ).write( json_string )
            model.save_weights( model_save_prefix + ".h5", overwrite = True )
            errw( "Done!\n" )

        errw( "\t\tepoch finished at: " + str( datetime.datetime.now() ) + "\n" )


# returns:
#   - amino acid accuracy
#   - codon accuracy
#   - amino acid confusion matrix
#   - codon confusion matrix
def calc_category_accuracy( all_predictions, all_labels, idx_to_codon, epoch_counter, model_save_prefix ):
    counter = 0
    cor_cds = 0
    err_cds = 0
    cor_aa = 0
    err_aa = 0

    # confusion matrices
    ## first key is the correct value while the second key is the incorrectly predicted value
    ## amino acid confusion matrix
    #inner_aa_cf = { item : 0 for item in all_aas }
    #aa_cf = { item : copy.deepcopy( inner_aa_cf ) for item in all_aas }
    aa_cf = pd.DataFrame( index = all_aas, columns = all_aas )
    aa_cf.fillna( value = 0, inplace = True )

    ## codon confusion matrix
    #inner_codon_cf = { item : 0 for item in all_codons }
    #codon_cf = { item : copy.deepcopy( inner_codon_cf ) for item in all_codons }
    codon_cf = pd.DataFrame( index = all_codons, columns = all_codons )
    codon_cf.fillna( value = 0, inplace = True )

    aas_out = open( model_save_prefix + "." + str( epoch_counter ) + ".aas_test", 'w' )
    cds_out = open( model_save_prefix + "." + str( epoch_counter ) + ".cds_test", 'w' )

    for seq_predictions, seq_labels in zip( all_predictions, all_labels ):
        # first form CDS
        # then translate, then count
        # we have to translate the predicted sequence because it might output an invalid sequence
        # and biopython does not like that.
        pred_cds = []
        true_cds = []
        pos = 0
        count = []

        for pred_val, true_val in zip( seq_predictions, seq_labels ):
            pred_lab = np.argmax( pred_val )
            true_lab = np.argmax( true_val )
    
            if true_lab == 0:
                break

            # fix this to account for the amino acid U
            true_codon = idx_to_codon[ true_lab ]
            pred_codon = idx_to_codon[ pred_lab ]

            true_cds.append( true_codon )
            pred_cds.append( pred_codon )

            #true_aa = codon_to_aa[ true_codon ]
            #pred_aa = codon_to_aa[ pred_codon ]

            if true_codon == pred_codon:
                cor_cds += 1
                count.append( 0 )
            else:
                err_cds += 1
                count.append( 1 )
                #codon_cf[ true_lab ][ pred_lab ] += 1
                codon_cf[ true_codon ].loc[ pred_codon ] += 1
        true_cds = "".join( true_cds )
        pred_aas = translate( "".join( pred_cds ), table = 11 )

        cds_out.write( ">" + str( counter ) + "\n" )
        cds_out.write( "  ".join( map( str, count ) ) + "\n" )
        cds_out.write( "".join( true_cds ) + "\n" )
        cds_out.write( "".join( pred_cds ) + "\n" )

        try:
            true_aas = translate( true_cds, table = 11, cds = True ) + "*"
        except:
            true_aas = translate( true_cds, table = 11 )

        aas_out.write( ">" + str( counter ) + "\n" )
        aas_out.write( "".join( map( str, count ) ) + "\n" )
        aas_out.write( true_aas + "\n" )
        aas_out.write( pred_aas + "\n" )

        for true_aa, pred_aa, count_me in zip( true_aas, pred_aas, count ):
            if count_me == 0:
                cor_aa += 1
                continue

            if true_aa == pred_aa:
                cor_aa += 1
            else:
                err_aa += 1
                #aa_cf[ true_codon ][ pred_codon ] += 1
                aa_cf[ true_aa ].loc[ pred_aa ] += 1

        counter += 1
    aas_out.close()
    cds_out.close()

    # this will return the following values to measure accuracy:
    # ( amino acid accuracy, codon accuracy )
    return float( cor_aa ) / ( cor_aa + err_aa ), float( cor_cds ) / ( cor_cds + err_cds ), aa_cf, codon_cf


def get_accuracy( outputs, labels, idx_to_codon ):
    gen_seqs = []
    cor_seqs = []
    correct = 0.
    incorrect = 0.

    for predicted, actual in zip( outputs, labels ):
        gen_seq = ""
        cor_seq = ""
        for idx, pred in enumerate( predicted ):
            codon_idx = np.argmax( pred )
            codon_prob = np.amax( pred )
            codon = idx_to_codon[ codon_idx ]

            cor_codon_idx = np.argmax( actual[ idx ] )
            cor_codon = idx_to_codon[ cor_codon_idx ]

            #errw( codon + "\t" + cor_codon + "\n" )
            if cor_codon == "":
                continue
            if cor_codon == codon:
                correct += 1.
            else:
                incorrect += 1.
            gen_seq += codon
            cor_seq += cor_codon
        #errw( gen_seq + "\n" )
        #errw( cor_seq + "\n" )

        gen_seqs.append( gen_seq )
        cor_seqs.append( cor_seq )

    accuracy = correct / ( correct + incorrect )
    return accuracy, gen_seqs, cor_seqs


# Take a trained model and print out the accuracy on the test set
def test_model( model, model_save_prefix, idx_to_codon, test_x, test_y, print_test_seqs ):
    # show accuracy
    # this method doesn't seem to exist for graph models, it does for
    # sequential models though... which is odd
    errw( "Testing model\n" )
   
    errw( "\tClassifying test data set..." )
    # generate predictions
    results = model.predict(
            {
                "input" : test_x,
                "output" : test_y
            },
            verbose = 1
            )
    errw( "Done!\n" )

    errw( "\tComputing accuracy..." )
    outputs = results[ "output" ]
    
    accuracy, gen_seqs, cor_seqs = get_accuracy( outputs, test_y, idx_to_codon )

    if print_test_seqs:
        fh = open( "test_dataset" + model_save_prefix +  ".txt", 'w' )
        
        counter = 0
        for i in range( len( gen_seqs ) ):
            fh.write( ">predicted" + str( counter ) + "\n" )
            fh.write( gen_seqs[ i ] + "\n" )
            fh.write( ">correct" + str( counter ) + "\n" )
            fh.write( cor_seqs[ i ] + "\n" )
        counter += 1
        
        fh.close()

    errw( "Done!\n" )
    errw( "\tTest set accuracy: " + str( accuracy ) + "\n" )


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
    start = time.time()

    # reset the model save path to include information about your network
    args.model_save_path += '.' + str( args.embedding_nodes ) + '.' + str( args.hidden_layers ) + '.' + str( args.lstm_nodes )

    if args.gru:
        args.model_save_path += ".gru"

    errw( "Detrans " + version + "\n\n" )
    errw( "Start date/time: " + str( datetime.datetime.now() ) + "\n" )
    errw( "Input parameters:\n" )
    errw( "\tAmino acid file: " + args.amino_acids_path + "\n" )
    errw( "\tCodons file: " + args.codons_path + "\n" )
    errw( "\tNumber epochs for training: " + str( args.epochs ) + "\n" )

    if not args.load_model:
        errw( "\tHidden layers: " + str( args.hidden_layers ) + "\n" )
        errw( "\tEmbedding layer nodes: " + str( args.embedding_nodes ) + "\n" )
        errw( "\tRNN output nodes: " + str( args.lstm_nodes ) + "\n" )
        errw( "\tData splits: " + args.training_split + "\n" )
        errw( "\tModel training verbosity level: " + str( args.verbosity ) + "\n" )
        errw( "\tDropout W: " + str( args.dropout_w ) + "\n" )
        errw( "\tDropout U: " + str( args.dropout_u ) + "\n" )
        if args.gru:
            errw( "\tUsing GRU in the RNN\n" )
            node_type = GRU
        else:
            errw( "\tUsing LSTM in the RNN\n" )
            node_type = LSTM
        if args.forwards_only:
            errw( "\tForwards only network (not bidirectional)\n" )

    else:
        errw( "\tModel parameters from saved model\n" )

    if args.print_test_seqs:
        errw( "Printing out the test set sequences after training\n" )
    
    errw( "\tMaximum sequence length for training: " )
    if args.seq_len_cutoff == 0:
        errw( "Full length sequences\n" )
    else:
        errw( str( args.seq_len_cutoff ) + "\n" )

    errw( "\tNumber of seqs for train, validate, test: " )
    if args.max_seqs == 0:
        errw( "All\n" )
    else:
        errw( str( args.max_seqs ) + "\n" )
    
    if args.classify:
        errw( "\tClassifying: " + args.clasify + "\n" )
    
    if args.load_model:
        errw( "\tLoading model from: " + args.load_model + "\n" )
        args.model_save_path = args.load_model
        if args.one_shot:
            args.model_save_path += "." + args.one_shot

    if args.no_save:
        errw( "\tNot saving model architecture and weights\n" )
    else:
        errw( "\tModel save path prefix: " + args.model_save_path + "\n" )

    if args.seed:
        errw( "\tSetting data shuffle random seed to: " + str( args.seed ) + "\n" )

    if args.one_shot:
        errw( "\tPerforming one-shot training.\n" )
        

    # load data
    aa_index, aa_seqs, cds_index, cds_seqs = load_data( args.amino_acids_path, args.codons_path, args.seq_len_cutoff )

    # prepare needed parameters for model training
    aa_vocab_size = len( aa_index )
    cds_vocab_size = len( cds_index )
    max_seq_len = ( len( max( aa_seqs, key = len ) ) + 1 ) / 2

    errw( "Amino acid vocab size: " + str( aa_vocab_size ) + "\n" )
    errw( "Codons vocab size: " + str( cds_vocab_size ) + "\n" )
    errw( "Maximum sequence length: " + str( max_seq_len ) + "\n" )

    # check if we're loading a previously trained model
    if args.load_model:

        # load in parameters and indices
        model, t_aa_index, t_cds_index = load_model( args.load_model, args.one_shot )
        
        # Perform a check to make sure the loaded vocabulary from a previous run contains all words for the currently laoded dataset
        if not set( aa_index.keys() ).issubset( set( t_aa_index.keys() ) ) or not set( cds_index.keys() ).issubset( set( t_cds_index.keys() ) ):
            sys.exit( "ERROR: loaded amino acid index does not contain all words in the loaded traning set. Aborting!\n" )

        # if tests pass then we need to overwrite the indices
        aa_index = t_aa_index
        cds_index = t_cds_index

        # if we're doing one shot learning, we need to modify the loaded model so that only the last layer is trainable.[ 0 ]
    else:
        # save the indices for loading models later
        pickle.dump( aa_index, open( args.model_save_path + ".aa_index.p", "wb" ) )
        pickle.dump( cds_index, open( args.model_save_path + ".cds_index.p", "wb" ) )

        model = build_model( args.hidden_layers, args.embedding_nodes, args.lstm_nodes, aa_vocab_size, cds_vocab_size, max_seq_len, args.forwards_only, node_type, args.dropout_w, args.dropout_u )


    errw( "Shuffling data..." )
    # randomize the data, using a seed if necessary
    if args.seed:
        errw( "using seed " + str( args.seed ) + "..." )
        aa_seqs, cds_seqs = shuffle( aa_seqs, cds_seqs, random_state = args.seed )
    else:
        aa_seqs, cds_seqs = shuffle( aa_seqs, cds_seqs )
    errw( "Done!\n" )

    errw( "Total number of instances read: " + str( len( aa_seqs ) ) + "\n" )
    if args.max_seqs > 0:
        errw( "\tOnly using " + str( args.max_seqs ) + " sequences\n" )
        aa_seqs = aa_seqs[ : args.max_seqs ]
        cds_seqs = cds_seqs[ : args.max_seqs ]
   
    cds_reverse_index = number_to_word( cds_index )

    # prepare text for model training
    errw( "Prepare sequences for input into learning algorithms\n" )
    
    errw( "\tConvert amino acid sequences..." )
    aa_seqs = seqs_to_indices( aa_seqs, aa_index )
    errw( "Done!\n" )
    
    errw( "\tPad amino acid sequences..." )
    # our convention for padding is to have padding on the right of the sequence,
    # this is reflected in the seqs_to_one_hot function
    aa_seqs = pad_sequences( aa_seqs, maxlen = max_seq_len, padding = 'post' )
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

    #print validate_x[ 0 ]
    #print validate_y[ 0 ]
    #print cds_seqs

    # train the model
    train( model, train_x, train_y, validate_x, validate_y, args.epochs, args.verbosity, args.model_save_path, cds_reverse_index, args.no_save, args.save_confusion_matrix )

    # run model on test dataset and print accuracy
    #test_model( model, args.model_save_path, cds_reverse_index, test_x, test_y, args.print_test_seqs )

    # check if we need to classify another external file
    # classify if we need to and output the results to a file
    if args.classify:
        classify( model, args.classify )

    errw( "Done!\n" )
    end = time.time()

    errw( "End date/time: " + str( datetime.datetime.now() ) + "\n" )
    print_runtime( start, end )

    errw( "Done!\n" )


def print_runtime( start, end ):
    # in seconds
    runtime = int( end - start )
    
    days = runtime / ( 60 * 60 * 24 )
    runtime -= days * ( 60 * 60 * 24 )

    hours = runtime / ( 60 * 60 )
    runtime -= hours * ( 60 * 60 )

    mins = runtime / ( 60 )
    runtime -= mins * 60
    
    secs = runtime

    # format strings
    days = str( days )
    days = "0" * ( 2 - len( days ) ) + days

    hours = str( hours )
    hours = "0" * ( 2 - len( hours ) ) + hours

    mins = str( mins )
    mins = "0" * ( 2 - len( mins ) ) + mins

    secs = str( secs )
    secs = "0" * ( 2 - len( secs ) ) + secs

    errw( "Total runtime: " + days + ":" + hours + ":" + mins + ":" + secs + "\n" )
    

if __name__ == "__main__":
    errw( "Command run: " + " ".join( sys.argv ) + "\n" )
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
            action = 'store_true',
            help = "Use only forwards LSTMs (Default False)."
            )
    parser.add_argument( '--verbosity',
            type = int,
            help = "Verbosity level when training the network (Default 0)."
            )
    default_model_save_path = "model." + time.strftime( "%Y-%m-%d" ) + "-" + time.strftime( "%H-%M-%S" )
    parser.add_argument( '--seq_len_cutoff',
            type = int,
            help = "The maximum sequence length that will be used in training. If it is longer than this, it will be ignored. Use 0 to allow all lengths (Default 0 )."
            )
    parser.add_argument( '--load_model',
            type = str,
            help = "Prefix for the model that you want to load. The prefix should have a corresponding .json and .h5 file."
            )
    parser.add_argument( '--max_seqs',
            type = int,
            help = "The maximum number of sequences to use in training, validation, and test. Specify 0 for all (Default 0)."
            )
    parser.add_argument( '--print_test_seqs',
            action = 'store_true',
            help = "Print out the test dataset predictions as well as the correct sequence to a file (Default False)."
            )
    parser.add_argument( '--no_save',
            action = 'store_true',
            help = "Don't save the model parameters (Default False)."
            )
    parser.add_argument( '--seed',
            type = int,
            help = "Random seed to use when shuffling data."
            )
    parser.add_argument( '--gru',
            action = 'store_true',
            help = "Change from the default LSTM to using a GRU. GRU may train faster than LSTM."
            )
    parser.add_argument( '--dropout_w',
            type = float,
            help = "Float between 0 and 1. Fraction of input units to drop for input gates (Default 0)."
            )
    parser.add_argument( '--dropout_u',
            type = float,
            help = "Float between 0 and 1. Fraction of input units to drop for recurrent connections (Default 0)."
            )
    parser.add_argument( '--one_shot',
            type = str,
            help = "Use one-shot learning, add identifier to the trained model so the pre-trained model is preserved and the fine-tuned model is saved to a new location."
            )
    parser.add_argument( '--save_confusion_matrix',
            action = 'store_true',
            help = 'Save the confusion matrices for both codons and amino acids.'
            )
    parser.set_defaults(
            save_confusion_matrix = False,
            hidden_layers = 1,
            embedding_nodes = 128,
            lstm_nodes = 128,
            model_save_path = default_model_save_path,
            epochs = 5,
            training_split = "70,15,15",
            forwards_only = False,
            verbosity = 0,
            seq_len_cutoff = 0,
            max_seqs = 0,
            print_test_seqs = False,
            no_save = False,
            seed = None,
            gru = False,
            dropout_w = 0.0,
            dropout_u = 0.0
            )

    args = parser.parse_args()

    # *******************************************
    # argument dependency validation

    ## This is 
    if args.one_shot and not args.load_model:
        sys.exit( "ERROR: one shot learning was specified but no pre-trained model was specified to load.\n\tUse the --load_model flag to specify a pre-trained model.\n\tAborting!\n" )

    # end argument dependency validation
    # *******************************************

    # *******************************************
    # argument parameter validation
    split_total = sum( map( float, args.training_split.split( ',' ) ) )
    if split_total > 100.1 or split_total < 99.9:
        sys.exit( "ERROR: Dataset split " + args.training_split + " does not sum to 100!\n\tAborting!\n" )
    
    # end argument parameter validation
    # *******************************************

    main( args )


