'''
This script is designed to take in a trained model dataset for classification and classify it.
No training is possible with this script.

modified by stanley fujimoto (masakistan)
16 mar 2016
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


# Shorthand way to print to standard error
def errw( string ):
    sys.stderr.write( string )


def main( args ):
    pass


if __name__ == "__main__":
    errw( "Command: " + " ".join( sys.argv ) + "\n" )
    parser = arpgarse.ArgumentParser(
            description = "Detranslate a set of sequences in fasta or nlp format given a trained model."
            )
    parser.add_argument( 'model_prefix',
            type = str,
            description = "Prefix for the model to be loaded."
            )
    parser.add_argument( '--fasta',
            type = str,
            description = "Path to the a fasta file for detranslation."
            )
    parser.add_argument( '--nlp',
            type = str,
            description = "Path to the nlp formatted file for detranslation."
            )

    args = parser.parse_args()

    main( args )

