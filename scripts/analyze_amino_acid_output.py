import sys, argparse


def main( args ):
    with open( args.predicted ) as fh:
        for line in fh:
            if line[ 0 ] == '>':
                name = line.strip()
                marker = fh.next().strip()
                t_seq = fh.next().strip()
                p_seq = fh.next().strip()

                # analyze
                idx = 0
                for mark, t, p in zip( marker, t_seq, p_seq ):
                    if mark == '1':
                        if t != p:
                            print name, "\t", str( idx )
                            print marker
                            print t_seq
                            print p_seq
                            break
                    idx += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Analyze stuff?"
            )
    parser.add_argument( "predicted",
            help = "Input file that contains the predicted and true output to be analyzed",
            type = str
            )
    args = parser.parse_args()
    main( args )

