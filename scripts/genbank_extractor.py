import argparse, sys, re
from Bio.Seq import Seq
from Bio import SeqIO


def main( args ):
    print "processing " + args.gb + " ..."
    cds_out = open( args.gb + ".cds.fasta", 'w' )
    aas_out = open( args.gb + ".aas.fasta", 'w' )
    gb_file = open( args.gb, "rU" )
    counter = 0
    printed_seqs = 0
    t11 = 0
    tother = 0
    for rec in SeqIO.parse( gb_file, "genbank" ):
        #print rec.seq
        #aa = rec.seq.translate( cds = True )
        #print aa
        #sys.exit()
        if rec.features:
            for feat in rec.features:
                if feat.type == "CDS":
                    counter += 1
                    name = None
                    aa = None
                    nuc = None
                    table = None
                    nuc_seq = None
                    try:
                        #print feat.qualifiers[ "protein_id" ]
                        #print feat.qualifiers[ "translation" ]
                        name = feat.qualifiers[ "protein_id" ][ 0 ]
                        aa = feat.qualifiers[ "translation" ][ 0 ]
                        nuc = str( feat.location.extract( rec ).seq )
                        nuc_seq = feat.location.extract( rec ).seq
                        table = feat.qualifiers[ "transl_table" ][ 0 ]
                        #print feat.qualifiers[ "protein_id"]
                        #print feat.qualifiers[ "translation" ]
                        #print nuc
                    except:
                        #print "invalid!*********************************************"
                        #nuc = ""
                        pass

                    if nuc is not None:
                        if table == "11":
                            t11 += 1
                        else:
                            tother += 1

                        if not re.match( '^[ACGT]+$', nuc ):
                            continue
                        printed_seqs += 1
                        cds_out.write( ">" + name + "\n" )
                        cds_out.write( nuc + "\n" )
                        aas_out.write( ">" + name + "\n" )
                        aas_out.write( aa + "*\n" )
                    #print feat.location
                    #print feat.qualifiers
                    #print feat.qualifiers[ "protein_id"]
                    #print feat.location.extract( rec ).seq
    gb_file.close()
    cds_out.close()
    aas_out.close()
    print "\tprocessed " + str( counter ) + " coding sequences."
    print "\t\tprinted " + str( printed_seqs ) + " valid sequences to fasta files."
    print "\t\tskipped " + str( counter - printed_seqs ) + " sequences."
    print "\t\tcodon table 11: " + str( t11 )
    print "\t\tother codon table: " + str( tother )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Translate a genbank file into two separate fasta files, one with the nucleotide CDS and the other with amino acids."
            )
    parser.add_argument( "--gb",
            required = True,
            type = str,
            help = "Input genbank file."
            )
    args = parser.parse_args()
    main( args )
