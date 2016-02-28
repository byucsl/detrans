'''
This script is designed to take a list of NC_ IDs from NCBI and download the coding sequence for them from NCBI via entrez.

written by stanley fujimoto (masakistan)
27 feb 2016
'''

import sys, argparse, time
from Bio import Entrez

def read_accessions( file_path ):
    ids = []

    with open( file_path, 'r' ) as fh:
        for line in fh:
            ids.append( line.strip() )

    return ids

def request_cds( org_id, email ):
    Entrez.email = email
    handle = Entrez.efetch(db="sequences", id=org_id, rettype="fasta_cds_na", retmode="text")
    text = handle.read()
    return text

def write( path, text ):
    with open( path, 'w' ) as fh:
        fh.write( text )

def make_queries( output_dir, email, ids ):
    for id in ids:
        sys.stderr.write( "\tQuerying " + id + "..." )
        
        start = time.time()

        fasta_cds = request_cds( id, email )
        path_cds = output_dir + "/" + id + ".cds.fasta"
        write( path_cds, fasta_cds )

        end = time.time()

        elapsed_time = end - start
        
        sys.stderr.write( "Done! (" + str( elapsed_time ) + " secs)\n" )

def main( args ):
    sys.stderr.write( "Reading IDs from: " + args.input_list + "\n" )
    ids = read_accessions( args.input_list )
    sys.stderr.write( "\tRead " + str( len( ids ) ) + " ids.\n" )

    sys.stderr.write( "Making entrez queries...\n" )
    make_queries( args.output_dir, args.email, ids )
    sys.stderr.write( "Finished!\n" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Utilize entrez to download coding sequences from NCBI"
            )
    parser.add_argument( 'input_list',
            type = str,
            help = "List of NC_ accession numbers, one per line."
            )
    parser.add_argument( 'output_dir',
            type = str,
            help = "Directory to write CDS to."
            )
    parser.add_argument( 'email',
            type = str,
            help = "NCBI requires an email address to use entrez queries."
            )
    
    args = parser.parse_args()

    main( args )
