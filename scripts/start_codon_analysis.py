'''
pass codons first then amino acids
'''


import sys


trans = {}

codon = []
amino = []
with open( sys.argv[ 1 ], 'r' ) as fh:
    for line in fh:
        if len( line.strip() ) == 0:
            continue
        codon.append( line.strip().split()[ 0 ] )
with open( sys.argv[ 2 ], 'r' ) as fh:
    for line in fh:
        if len( line.strip() ) == 0:
            continue
        amino.append( line.strip().split()[ 0 ] )

for a, c in zip( amino, codon ):
    try:
        trans[ c ].add( a )
    except:
        trans[ c ] = set( a )

print trans.keys()

for key in sorted( trans.keys() ):
    val = trans[ key ]
    print key, ":", ", ".join( val )
