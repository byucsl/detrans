# detrans

## Purpose

This project is designed to take an amino acid sequence and convert it into a nucleotide sequence.
The higher level purpose of this project is to create a different view of codon bias and how we model it.
Here, we attempt to model codon bias as a natural language processing problem, specifically, translation.
Generating a language model that can then be used to translate one amino acid sequence string to a nucleotide sequence string provides a method for genering sequences that can be inserted into vectors that are optimized for the codon bias of a particular organism.

## Methods

We utilized recurrent neural networks (RNNs) in order to accomplish this.
Specifically, we use Long Short-Term Memory (LSTM) networks as a way to learn nucleotide sequence encodings of arbitrary length amino acid sequences.
LSTMs provide a way to model the long term dependencies that occur within sequences.
Training of networks is best accomplished using GPUs as they provide significant speedup for network training.

## Workflow

1. Gather coding DNA sequences (CDS) for training for many species related to your vector
1. Train network using these sequences
1. Select a specific species (the vector you will be using), select only their highly expressed genes
1. Fine tune the network trained on many species with many species by using the subset of highly expressed genes for your vector

See the below for a more comprehensive explanation of steps including different scripts to use for different aspects of the workflow.

1. Fetch and prepare CDS data for training
  1. Create a text file that contains the NCBI genome IDs for species you will train with
  1. Use [entrez_fetch_genome.py] (scripts/entrez_fetch_genome.py) to fetch genomes from NCBI.
  1. Use [entrez_fetch_ft.py] (scripts/entrez_fetch_ft.py) to fetch the feature tables for the specified species from NCBI.
  1. Use [extract_cds_from_fasta.py] (scripts/extract_cds_from_fasta.py) to extract the CDS from genomes using the feature tables. This will also create a fasta file containing the translated sequences. Note that this script will remove any sequences that contain ambiguity codes. Ambiguity codes are not supported by this application at this time.
  1. Use [fasta_nlp.py] (scripts/fasta_nlp.py) to convert the fasta CDS and amino acid sequence files to the correct format for training.
1. Train the network
  1. Use [detrans_train.py] (networks/detrans_train.py) to train the general network
1. Prepare data for one-shot learning
  1. Select species of interest (most likely the vector you're using).
  1. Extract CDS for selected species.
  1. Filter CDS, keep only highly expressed (or other characteristic) sequences.
1. One-shot learning
  1. use [detrans_one_shot.py] (networks/detrans_one_shot.py) to fine tune your network for your specific vector
1. Detranslate proteins
  1. Use [detrans_classify.py] (networks/detrans_classify.py) to generate a nucleotide sequence from a polypeptide.

## Tutorial

Use the following steps for an end to end example on how to run detrans.

```bash
# Install dependencies, it is suggested you use virtualenv
pip install -r requirements.txt

# Create list of genomes to use for training

# Fetch and prepare CDS data for training
scripts/entrez_fetch_genome.py args...
scripts/entrez_fetch_ft.py args...
scripts/extract_cds_from_fasta.py args..

# Format data for training
scripts/fasta_nlp.py args...

# Train the model
networks/detrans_train.py args...

# Prepare data for one-shot learning
scripts/fasta_nlp.py args...

# One-shot learning
networks/detrains_train.py --one_shot --load_model model_prefix args...

# Detranslate sequences of interest
```

## Tips for running

1. You may consider training with the --gru option to use GRUs instead of LSTMs as GRUs may train faster
1. During one-shot training, use 1 (or few) training epochs so that you don't overtrain your models

## Dependencies

1. keras (https://github.com/fchollet/keras)
  1. Make sure that you're running the newest version of keras (pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps)
1. theano (https://github.com/Theano/Theano)
1. scikit-learn (https://github.com/scikit-learn/scikit-learn)
1. h5py

## Publication

## Acknowledgements

The authors would like to thank the following individuals for their support in developing this project:

1. BYU Deep-Learning Study Group
    1. Mike Brodie
    1. Aaron Dennis
    1. Derrall Heath
    1. Logan Mitchell
    1. Christopher Tensmeyer
1. Alexander Lemon
1. BYU Computational Science Laboratory

## Contributors
@masakistan (sfujimoto@gmail.com)
