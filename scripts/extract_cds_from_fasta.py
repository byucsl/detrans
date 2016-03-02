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


codon_to_aa = {"TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
    "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"*", "TAG":"*",
    "TGT":"C", "TGC":"C", "TGA":"*", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L",
    "CCT":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAT":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGT":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "ATT":"I", "ATC":"I", "ATA":"I", "ATG":"M",
    "ACT":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAT":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGT":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GTT":"V", "GTC":"V", "GTA":"V", "GTG":"V",
    "GCT":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAT":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G"}


def translate( seq ):
    translated_seq = ""
    codons = [ seq[ i : i + 3 ] for i in range( 0, len( seq ), 3 ) ]
    for idx, codon in enumerate( [ seq[ i : i + 3 ] for i in range( 0, len( seq ), 3 ) ] ):
        try:
            translated_seq += codon_to_aa[ codon ]
        except KeyError:
            return ""
    return translated_seq


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

def extract_cds( genome, ft, full_feature_names ):
    sys.stderr.write( "\tExtracting coding sequences..." )
    cds = []
    aas = []
    names = []
    skipped_ambiguous_seqs = 0
    skipped_pseudo_seqs = 0
    skipped_mystery_seqs = 0

    for f in ft:
        start = f[ 0 ][ 0 ]
        end = f[ 0 ][ 1 ]

        seq_name = ""
        pseudo = False
        
        for attrib in f[ 1 ]:
            if attrib[ 0 ] == 'note' and not full_feature_names:
                continue
            if attrib[ 0 ] == 'pseudo':
                pseudo = True
                break
            seq_name += "[" + attrib[ 0 ]
            if len( attrib ) > 1:
                seq_name += "=" + attrib[ 1 ] + "] "
            else:
                seq_name += "] "

        if pseudo:
            skipped_pseudo_seqs += 1
            continue


        if start > end:
            #print "*" * 100
            #print seq_name
            #print seq
            #print "*" * 100
            seq = rev_com( genome[ end - 1 : start ] )
        else:
            seq = genome[ start - 1 : end ]

        trans_seq = translate( seq )
        if trans_seq == "":
            skipped_ambiguous_seqs += 1
            continue
        if seq_name == "":
            skipped_mystery_seqs += 1
            continue

        names.append( seq_name )
        cds.append( seq )
        aas.append( trans_seq )

    sys.stderr.write( " Done!\n" )
    if skipped_pseudo_seqs > 0 or skipped_ambiguous_seqs > 0 or skipped_mystery_seqs > 0:
        sys.stderr.write( "\t\tskipped:\n" )
        sys.stderr.write( "\t\t\t" + str( skipped_pseudo_seqs ) + " pseudo genes\n" )
        sys.stderr.write( "\t\t\t" + str( skipped_ambiguous_seqs ) + " seqs with ambiguity codes\n" )
        sys.stderr.write( "\t\t\t" + str( skipped_mystery_seqs ) + " seqs with no feature attributes\n" )

    return names, cds, aas


def print_cds( names, cds, aas, print_aa, cds_out_file, aa_out_file ):
    sys.stderr.write( "\tPrinting coding sequence" )
    if print_aa:
        sys.stderr.write( " and amino acid sequences" )
        aa_fh = open( aa_out_file, 'w' )
    sys.stderr.write( "..." )

    erroneous_aas = 0

    with open( cds_out_file, 'w' ) as fh:
        for idx, seq in enumerate( cds ):
            fh.write( ">" + names[ idx ] + "\n" )
            fh.write( seq + "\n" )

            if print_aa:
                aa_fh.write( ">" + names[ idx ] + "\n" )
                aa_fh.write( aas[ idx ] + "\n" )
                
                # sanity check to make sure that all amino acid sequences end with
                # an asterisk (end of seq char)
                if aas[ idx ][ -1 ] != "*":
                    erroneous_aas += 1

    if print_aa:
        aa_fh.close()

    sys.stderr.write( " Done!" )
    if erroneous_aas > 0:
        sys.stderr.write( " (" + str( erroneous_aas ) + " amino acid seqs that don't end with *)" )
    sys.stderr.write( "\n" )


def main( args ):
    sys.stderr.write( "Genome: " + args.fasta + "\n" )
    sys.stderr.write( "Feature table: " + args.ft + "\n" )
    cds_out_file = args.fasta + ".cds.fa"
    aa_out_file = args.fasta + ".aa.fa"

    sys.stderr.write( "CDS output file: " + cds_out_file + "\n" )
    if args.print_aa:
        sys.stderr.write( "AA output file: " + aa_out_file + "\n" )

    genome = parse_genome( args.fasta )
    ft = parse_ft( args.ft )

    names, cds, aas = extract_cds( genome, ft, args.full_feature_names )
    print_cds( names, cds, aas, args.print_aa, cds_out_file, aa_out_file )

    sys.stderr.write( "Done!\n" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "This tool takes in a fasta file and a features table as downloaded from NCBI via entrez and extracts only the coding sequences and outputs it to a fasta file. This only works for bacterial genomes. Outputs a fasta file with the extracted features with a \'.cds.fa\' extension. Optionally, it can also translate these sequences and output a fasta file with a \'.aa.fa\' extension." )
    parser.add_argument( 'fasta',
            type = str,
            help = "Fasta file containing an organisms genome."
            )
    parser.add_argument( 'ft',
            type = str,
            help = "Feature table as downloaded from NCBI via entrez."
            )
    parser.add_argument( '--full_feature_names',
            action = 'store_true',
            help = "Print the full feature table as sequence header. (Default False)."
            )
    parser.add_argument( '--print_aa',
            action = 'store_true',
            help = "Print a fasta file with the amino acid sequences"
            )
    parser.set_defaults(
            full_feature_names = False,
            print_aa = False
            )
    
    args = parser.parse_args()
    main( args )
