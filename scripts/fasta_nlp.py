'''
This script is designed to take fasta files as input and remove headers and put each sequence on one line.

written by stanley fujimoto (masakistan)
27 feb 2016
'''

import sys, argparse


def split_string_into_words( seq, word_size, overlap ):
    if overlap:
        text = " ".join( [ seq[ i : i + word_size ] for i in range( 0, len( seq ) - word_size + 1 ) ] )
    else:
        text = " ".join( [ seq[ i : i + word_size ] for i in range( 0, len( seq ), word_size ) ] )
    return text


def parse_and_print( input_file, output_file, word_size, overlap ):
    counter = 0
    with open( output_file, 'w' ) as fh_out:
        with open( input_file, 'r' ) as fh_in:
            seq = ""
            for line in fh_in:
                if line[ 0 ] == '>':
                    if seq != "":
                        formatted_str = split_string_into_words( seq, word_size, overlap )
                        fh_out.write( formatted_str + "\n" )
                        counter += 1
                        seq = ""
                else:
                    seq += line.strip()
            formatted_str = split_string_into_words( seq, word_size, overlap )
            fh_out.write( formatted_str + '\n' )
            if len( formatted_str ) > 0:
                counter += 1
    sys.stderr.write( "\tRead/Wrote " + str( counter ) + " seqs.\n" )
        

def main( args ):
    sys.stderr.write( "Reading from: " + args.input_file + "\n" )
    if args.output_file:
        output_path = args.output_file
        sys.stderr.write( "Writing to: " + args.output_path + "\n" )
    else:
        output_path = args.input_file + ".no_headers.txt"
        sys.stderr.write( "Writing to: " + output_path + "\n" )

    parse_and_print( args.input_file, output_path, args.word_size, args.overlap )

    sys.stderr.write( "Finished!\n" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = 'Input a fasta file, strip out headers, print sequence on one line.' )

    parser.add_argument( 'input_file',
            type = str,
            help = "Input fasta file."
            )
    parser.add_argument( '--output_file',
            type = str,
            help = "Output file path. If not provided, input file name will be used as prefix with \'.no_headers.txt\' extension added."
            )
    parser.add_argument( '--word_size',
            type = int,
            help = "The size of the words to separate the sequences into. 1 for amino acids, 3 for codons. etc. (Default 3)."
            )
    parser.add_argument( '--overlap',
            action = 'store_true',
            help = "Set this to allow words to overlap. Used for generating kmers as words." )
    parser.set_defaults( word_size = 3, overlap = False )

    args = parser.parse_args()

    main( args )
