import gensim.models as gsm

PATH_emoji2vec = ""
PATH_word2vec = ""

# load all emoji vectors from emoji2vec
e2v = gsm.KeyedVectors.load_word2vec_format(PATH_emoji2vec, binary=True)

# load all word vectors from word2vec 
w2v = gsm.KeyedVectors.load_word2vec_format(PATH_word2vec, binary=True)


# load sentence
sentence = "Mom deserves a Getaway. Happy Mothers Day"
sentence_words = sentence.split(" ")
wvecs = []
for word in sentence_words:
	vector = w2v[word]
	wvecs.append(vector)





# for all word vectors get cosine distance w.r.t emoji vectors

