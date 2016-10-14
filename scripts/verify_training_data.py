''' this is to verify the translation of the sequences.'''

import sys

nuc = open( sys.argv[ 1 ], 'r' )
aa = open( sys.argv[ 2 ], 'r' )

codon_to_aa = {
    "TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
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
    "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G"
    }


for nuc_line in nuc:
    nuc_line = nuc_line.split()
    aa_line = aa.readline().split()

    for nuc_codon, aa_a in zip( nuc_line, aa_line ):
        check_aa = codon_to_aa[ nuc_codon ]

        if check_aa != aa_a:
            print "Error! real codon: " + aa_a + "\ttranslated codon: " + check_aa



nuc.close()
aa.close()
