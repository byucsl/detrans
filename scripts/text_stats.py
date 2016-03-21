'''
This script will provide basic statistics on a given text file.
Each line is assumed to be a different sentence

Written by Stanley Fujimoto
'''

import sys

longest = 0
longest_seq = ''
shortest = float( 'inf' )
cur_sum = 0
count = 0

vocab = set()

with open( sys.argv[ 1 ] ) as fh:
    for line in fh:
        len_line = len( line.strip().split( ' ' ) )
        vocab.update( line.strip().split( ' ' ) )

        if len_line > longest:
            longest = len_line
            longest_seq = line.strip()
        if len_line < shortest:
            shortest = len_line
            shortest_seq = line.strip()
        cur_sum += len_line
        count += 1

print( "# of seqs:\t" + str( count ) )
print( "Longest length:\t" + str( longest ) )
print( "Shortest length:\t" + str( shortest ) )
print( "Average length:\t" + str( float( cur_sum ) / count ) )
print( "Vocabulary size:\t" + str( len( vocab ) ) )

'''print( "longest:" )
print( longest_seq )
print( "*************")
print( "shortest:" )
print( shortest_seq )'''

