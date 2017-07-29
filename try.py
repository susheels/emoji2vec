import gensim.models as gsm
import sklearn as sk
PATH_emoji2vec = "/home/susuresh/emoji2vec/pre-trained/emoji2vec.bin"
PATH_word2vec = "/home/susuresh/Downloads/GoogleNews-vectors-negative300.bin.gz"

# load all emoji vectors from emoji2vec
e2v = gsm.KeyedVectors.load_word2vec_format(PATH_emoji2vec, binary=True)

# load all word vectors from word2vec 
w2v = gsm.KeyedVectors.load_word2vec_format(PATH_word2vec, binary=True)


# load sentence
sentence = "Mom deserves a Getaway. Happy Mothers Day"
sentence_words = sentence.split(" ")
wvecs = []
for word in sentence_words:
	if word in w2v.vocab:
		vector = w2v[word]
		wvecs.append(vector)

for wvec in wvecs:
	print(e2v.similar_by_vector(wvec,topn=2))