'''
This script is designed to take a genomes table as provided by NCBI (e.g. http://www.ncbi.nlm.nih.gov/genome/genomes/167) and to strip out the NC_ ID's so that it can be used with entrez to download all the coding sequence for given ID.

IMPORTANT NOTES:
    - The table that you provide must be a csv, do not use the tab-delimmited file as they don't really use tabs and they're just spaces :(
    - This only designed to work for bacterial genomes that assum only one chromosome per organism.

TODO:
    - Extend for eukaryotes

written by stanley fujimoto (masakistan)
27 feb 2016
'''

import sys, argparse

def parse_table( input_file ):
    ids = []

    with open( input_file, 'r' ) as fh:
        for line in fh:
            line = line.strip().split( ',' )
    
            for seq_tup in line[ 10 ].split( ';' ):
                seq_tup = seq_tup.split( ':' )
                if seq_tup[ 0 ] == 'chromosome':
                    idx = seq_tup[ 1 ].find( '/' )
                    if idx != -1:
                        id = seq_tup[ 1 ][ : idx ]
                    else:
                        id = seq_tup[ 1 ]
                    ids.append( id )
    return ids

def write_table( output_file, ids ):

    with open( output_file, 'w' ) as fh:
        for id in ids:
            fh.write( id )
            fh.write( "\n" )


def main( args ):
    sys.stderr.write( "Parsing Genome Table: " + args.input_file + "\n" )
    sys.stderr.write( "Output file: " + args.output_file + "\n" )

    ids = parse_table( args.input_file )
    sys.stderr.write( "# of IDs read: " + str( len( ids ) ) + "\n" )
    
    write_table( args.output_file, ids )
    sys.stderr.write( 'IDs written to disk!\n' )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Parse the references genomes table comma separated file as provided by NCBI for a particular species (e.g. http://www.ncbi.nlm.nih.gov/genome/genomes/167) into a list that can be used with entrez for programmatic downloading of data."
            )
    parser.add_argument( 'input_file',
            type = str,
            help = "Input file that contains the Genomes Table from NCBI."
            )
    parser.add_argument( 'output_file',
            type = str,
            help = 'Output file to send list to.' )

    args = parser.parse_args()

    main( args )



