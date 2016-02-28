'''
the purpose of this script is to convert a multi-entry fasta file into a file
that is acceptable by the tensorflow translation program

writen by stanley fujimoto
11 dec 2015
'''

import sys, argparse

def convert( args ):
    input_path = args.input
    output_path = args.output
    word_size = args.word_size

    sys.stderr.write( "Reading from:\t" + input_path + "\n" )
    sys.stderr.write( "Writing to:\t" + output_path + "\n" )

    count = 0
    len_sum = 0
    shortest = float( 'inf' )
    longest = 0

    with open( input_path ) as in_f:
        with open( output_path, 'w' ) as out_f:
            seq = ""
            for line in in_f:
                if line[ 0 ] == '>':
                    if len( seq ) > 0:
                        out_f.write( " ".join( seq[ x * word_size : x * word_size + word_size ] for x in xrange( len( seq ) / word_size ) ) + "\n" )

                        if len( seq ) > longest:
                            longest = len( seq )
                        if len( seq ) < shortest:
                            shortest = len( seq )
                        len_sum += len( seq )

                        seq = ""
                        count += 1
                else:
                    seq += line.strip()

    sys.stderr.write( "Finished writing " + str( count ) + " sequences\n" )
    sys.stderr.write( "Longest sequence:\t" + str( longest ) + "\n" )
    sys.stderr.write( "Shortest sequence:\t" + str( shortest ) + "\n" )
    sys.stderr.write( "Average seq length:\t" + str( float( len_sum ) / count ) + "\n" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Convert a multi-entry fasta file into a format recognized by the Tensorflow translation program"
            )
    parser.add_argument( 'word_size',
            type = int,
            help = "Size of the word to split string into, all sequences must be divisible by the word_size"
            )
    parser.add_argument( 'input',
            type = str,
            help = "Input fasta file"
            )
    parser.add_argument( 'output',
            type = str,
            help = "Output file path"
            )

    args = parser.parse_args()

    convert( args )
