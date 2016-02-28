'''
This script is designed to take fasta files as input and remove headers and put each sequence on one line.

written by stanley fujimoto (masakistan)
27 feb 2016
'''

import sys, argparse

def parse_and_print( input_file, output_file ):
    counter = 0
    with open( output_file, 'w' ) as fh_out:
        with open( input_file, 'r' ) as fh_in:
            seq = ""
            for line in fh_in:
                if line[ 0 ] == '>':
                    if seq != "":
                        fh_out.write( seq + "\n" )
                        counter += 1
                        seq = ""
                else:
                    seq += line.strip()
            fh_out.write( seq )
            counter += 1
    sys.stderr.write( "Read/Wrote " + str( counter ) + " seqs.\n" )
        

def main( args ):
    sys.stderr.write( "Reading from: " + args.input_file + "\n" )
    if args.output_file:
        output_path = args.output_file
        sys.stderr.write( "Writing to: " + args.output_path + "\n" )
    else:
        output_path = args.input_file + ".no_headers.txt"
        sys.stderr.write( "Writing to: " + output_path + "\n" )

    parse_and_print( args.input_file, output_path )

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

    args = parser.parse_args()

    main( args )
