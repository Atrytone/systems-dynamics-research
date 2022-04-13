import pprint
from gensim import models
import json
import os

"""text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]"""


# Folder Path
path = "/home/atla/Documents/Programming/LDA practice/Documents"
  
# Change the directory
os.chdir(path)
  
# Read text File
  
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        val = f.read()
    return val
  

text_corpus = []  
print(len(text_corpus))

# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
  
        # call read text file function
        result = read_text_file(file_path)
        text_corpus.append(result)

print(len(text_corpus))


# Create a set of frequent words
stoplist = set('for a of the and to in is be are as on by with or will that this at an from it not'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]


# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
#print(dictionary)

pprint.pprint(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
#print(new_vec)

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
#pprint.pprint(bow_corpus)

model = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)
print(model)

print(model.print_topics(2))
model.save("./andre_and_heather_miraculous_model")