'''
This script is designed to take in a fasta genome file and to extract the coding sequences. This is designed to only work for bacterial genomes.

written by stanley fujimoto
1 mar 2016
'''

import sys, argparse
from string import maketrans

itab = "acgtACGT"
otab = "tgcaTGCA"
trantab = maketrans( itab, otab )

def rev_com( seq ):
    return seq.translate( trantab )[ :: -1 ]

def parse_genome( file_path ):
    sys.stderr.write( "\tParsing genome file..." )
    genome = ""
    with open( file_path ) as fh:
        seq_count = 0
        for line in fh:
            if line[ 0 ] == '>':
                seq_count += 1
                if seq_count > 1:
                    sys.stderr.write( "ERROR: More than one sequence in fasta file, this only works for bacterial genomes, can only have one sequence for a genome." )
                    sys.exit( 0 )
                continue
            genome += line.strip()
    sys.stderr.write( " Done!\n" )
    return genome

def parse_ft( file_path ):
    sys.stderr.write( "\tParsing feature table..." )
    ft = []
    skipped_seqs = 0
    with open( file_path ) as fh:
        cur_feature = []
        cur_attrib = []
        for line in fh:
            if line[ 0 ] == "\t":
                cur_attrib.append( line.strip().split( '\t' ) )
            elif line[ 0 ] == ">":
                continue
            else:
                if len( cur_feature ) > 0 and cur_feature[ -1 ] == 'CDS':
                    try:
                        cur_feature[ : -1 ] = map( int, cur_feature[ : -1 ] )
                        ft.append( [ cur_feature, cur_attrib ] )
                    except ValueError:
                        skipped_seqs += 1
                cur_feature = line.split()
                cur_attrib = []
    sys.stderr.write( " Done!" )
    if skipped_seqs > 0:
        sys.stderr.write( " (skipped " + str( skipped_seqs ) + " features)" )
    sys.stderr.write( "\n" )

    return ft

def extract_cds( genome, ft ):
    sys.stderr.write( "\tExtracting coding sequences..." )
    cds = []
    names = []

    for f in ft:
        start = f[ 0 ][ 0 ]
        end = f[ 0 ][ 1 ]

        seq_name = ""
        for attrib in f[ 1 ]:
            seq_name += "[" + attrib[ 0 ]
            if len( attrib ) > 1:
                seq_name += "=" + attrib[ 1 ] + "] "
            else:
                seq_name += "] "
        names.append( seq_name )

        if start > end:
            seq = rev_com( genome[ end - 1 : start ] )
        else:
            seq = genome[ start - 1 : end ]
        cds.append( seq )

    sys.stderr.write( " Done!\n" )
    return names, cds

def print_cds( names, cds, output_file ):
    sys.stderr.write( "\tPrinting coding sequence..." )
    with open( output_file, 'w' ) as fh:
        for idx, seq in enumerate( cds ):
            fh.write( ">" + names[ idx ] + "\n" )
            fh.write( seq + "\n" )
    sys.stderr.write( " Done!\n" )

def main( args ):
    sys.stderr.write( "Genome: " + args.fasta + "\n" )
    sys.stderr.write( "Feature table: " + args.ft + "\n" )
    output_file = args.fasta + ".cds.fa"

    sys.stderr.write( "Output file: " + output_file + "\n" )

    genome = parse_genome( args.fasta )
    ft = parse_ft( args.ft )

    names, cds = extract_cds( genome, ft )
    print_cds( names, cds, output_file )

    sys.stderr.write( "Done!\n" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "This tool takes in a fasta file and a features table as downloaded from NCBI via entrez and extracts only the coding sequences and outputs it to a fasta file. This only works for bacterial genomes." )
    parser.add_argument( 'fasta',
            type = str,
            help = "Fasta file containing an organisms genome."
            )
    parser.add_argument( 'ft',
            type = str,
            help = "Feature table as downloaded from NCBI via entrez."
            )
    
    args = parser.parse_args()
    main( args )
