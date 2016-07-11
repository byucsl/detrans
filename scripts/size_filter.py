'''
filter a formatted data file based on sequence length
'''

import sys, argparse


def main( args ):
    filtered = 0
    kept = 0
    with open( args.file, 'r' ) as fh:
        for line in fh:
            line = line.split()
            if len( line ) <= args.size:
                print " ".join( line )
                kept += 1
            else:
                filtered += 1

    sys.stderr.write( "Kept sequences:\t" + str( kept ) + "\nRemoved sequences:\t" + str( filtered ) + "\n" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Filter a formatted data file to remove sequences that are too big."
            )
    parser.add_argument( '--file',
            required = True,
            type = str,
            help = "The path to the file with the sequences."
            )
    parser.add_argument( '--size',
            required = True,
            type = int,
            help = "Sequences that are greater than this lenght will be excluded."
            )

    args = parser.parse_args()
    main( args )
