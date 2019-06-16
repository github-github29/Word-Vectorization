from __future__ import absolute_import, division,print_function
from nltk.corpus import stopwords
# for word encoding
import codecs
# for searching any file
import glob
import multiprocessing
import os
import re
import nltk
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def sentence_to_wordlist(raw):
	clean=re.sub("[^a-zA-Z]"," ",raw)
	words=clean.split()
	
	return words


book_filenames=sorted(glob.glob("./*.txt"))
print (book_filenames)
#one corpus 
corpus_raw=u""
for book_filename in book_filenames:
	print("Reading '{0}'...".format(book_filename))
	with codecs.open(book_filename,"r","utf-8") as book_file:
		corpus_raw+=book_file.read()
	print("corpus is now {0} coprus long".format(len(corpus_raw)))
	print()
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences=tokenizer.tokenize(corpus_raw)
 
 


sentences=[]
for raw_sentence in raw_sentences:
	if len(raw_sentence)>0:
		sentences.append(sentence_to_wordlist(raw_sentence))
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

token_count =sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

#train word2vec
num_features=300   #dimension
min_word_count=3   #threshold
num_workers=multiprocessing.cpu_count() 
context_size=7      #context window lenght (size of block at a time)
downsampling=1e-3   #frequent words
seed=1              #random number generator

harry2vec=w2v.Word2Vec(
	sg=1,
	seed=seed,
	workers=num_workers,
	size=num_features,
	min_count=min_word_count,
	window=context_size,
	sample=downsampling
)

harry2vec.build_vocab(sentences)
print("Word2Vec vocabulary length:",len(harry2vec.vocab))
harry2vec.train(sentences)



tsne = sklearn.manifold.TSNE(n_components=2,random_state=0)
all_word_vectors_matrix=harry2vec.syn0  #all vectors in one matrix
all_word_matrix_2d=tsne.fit_transform(all_word_vectors_matrix) #create 2d matrix
points=pd.DataFrame(							#dataframe:spreadsheet(rows and columns)
[
	(word,coords[0],coords[1])
	for word,coords in[
		(word,all_word_matrix_2d[harry2vec.vocab[word].index])
		for word in harry2vec.vocab
]
],
	columns=["word","x","y"]
)
print(points)



print(harry2vec.most_similar("Harry"))
plt.scatter(points.x,points.y)
plt.show()
-
