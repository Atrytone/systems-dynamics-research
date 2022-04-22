#!/usr/bin/python3
import pprint
from gensim import models
import json
import glob
import os
from collections import defaultdict
from gensim import corpora


def train_model(text_corpus):
    '''
    A function for training the actual LDA model.
    '''

    # Create a set of frequent words we want to remove.
    stoplist = set('for a of the and to in is be are as on by with or will that this at an from it not'.split(' '))

    # Conver each document to lowercase, split it by white space and filter out stopwords using list comprehension.
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in text_corpus]

    # Count word frequencies.
    frequency = defaultdict(int)

    for text in texts:

        for token in text:

            frequency[token] += 1

    # Only keep words that appear more than once.
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    pprint.pprint(processed_corpus)

    dictionary = corpora.Dictionary(processed_corpus)

    pprint.pprint(dictionary.token2id)

    new_doc = "Human computer interaction"
    new_vec = dictionary.doc2bow(new_doc.lower().split())

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    model = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)
    print(model)

    print(model.print_topics(2))
    model.save('./output_data/model')


def read_text_file(file_path):
    '''
    A function for reading in a single file.
    '''

    # Open the file and store a pointer in memory.
    with open(file_path, 'r') as f:

        try:
            # Read the file.
            data = f.read()

        # Some files throw decode errors because of characters. Catch these.
        except UnicodeDecodeError:

            print('There was an error decoding the file. Skipping...')

            # Since there was an error return nothing.
            return None

    # Return the file.
    return data


def ingest_data():
    '''
    A function for ingesting the relevant data before building the model.
    '''

    print('Creating list object for corpus...')

    # Create an empty list to hold the entire corpus and a counter for the files.
    text_corpus = []
    file_count = 0
    ingest_count = 0

    print('Enumerating over files...')

    # Iterate through all files
    #for file in os.listdir('./input_data/'):
    for file in glob.iglob(f'./input_data/*.txt'):

        file_count += 1

        print(f'Ingesting the following file: {file}')

        # Call read text file function to ingest the single file.
        result = read_text_file(f'{file}')

        # If a file is returned, add to the count and our corpus.
        if result is not None:

            ingest_count += 1

            print('File read. Appending to corpus list...')

            text_corpus.append(result)

    print(f'Ingested a total of {ingest_count} files out of {file_count} files...')

    return text_corpus


def main():
    '''
    The main function for orchestrating the python script.
    '''

    print('Ingesting data...')

    text_corpus = ingest_data()

    train_model(text_corpus)


if __name__ == '__main__':

    print('Initiating script...')

    main()
