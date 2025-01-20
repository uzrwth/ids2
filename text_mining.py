#!/usr/bin/env python

import multiprocessing
from IPython.display import display

import nltk
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')  # omw = open multilingual wordnet
stopword_list = set(stopwords.words('english'))
reg_tokenizer = nltk.RegexpTokenizer(r"\w+")  # tokenizes and removes punctuation at the same time
wordnet_lemmatizer = WordNetLemmatizer();


# (a) LOAD DATA


df = pd.read_csv("text_mining/emotions.csv")
value_cnts = df['label'].value_counts()
value_cnts_df = value_cnts.to_frame()

# k is the number of instances in the smallest sentiment label class
k = value_cnts.agg('min')

df2_index = pd.Index([], dtype='int64')
for i, g in df.groupby("label").groups.items():
	df2_index = df2_index.append(g[:k])


# balanced data
df2 = df.loc[df2_index].copy()
value_cnts_df2 = df2['label'].value_counts().to_frame()


# (b) PLOT


fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.barplot(data = value_cnts_df, x="count", y="label", ax=axes[0])
sns.barplot(data = value_cnts_df2, x="count", y="label", ax=axes[1])

axes[0].set_title("imbalanced data")
axes[1].set_title("balanced data")
plt.show()


# (c) PREPROCESS


def count_words(s=""):
	spaces  = sum(c.isspace() for c in s)
	return spaces + 1

def preprocess(s=""):
	s = s.lower()
	tokens = reg_tokenizer.tokenize(s)
	remove_stopword_tokens = [x for x in tokens if x not in stopword_list]
	lemm_tokens = map(lambda x: wordnet_lemmatizer.lemmatize(x), remove_stopword_tokens)
	return ' '.join(lemm_tokens)

df2["text2"] = df2["text"].map(preprocess)

before_token_cnt = df2["text"].transform(count_words).sum()
after_token_cnt = df2["text2"].transform(count_words).sum()

print("Token numbers(before v.s. after preprocessing): %d, %d" % (before_token_cnt, after_token_cnt))



# (d) SPLIT TRAIN AND TEST CORPUS

train_df2, test_df2 = train_test_split(df2, train_size=0.8, random_state=12345, stratify=df2["label"])

train_line = train_df2.iloc[2]["text"]
test_line = test_df2.iloc[2]["text"]
assert train_line=="i feel calm complete and whole after i meditate"
assert test_line=="i had a funny feeling when i accepted them"


# (e) TRAIN CLASSIFIER BoW, SGD


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df2["text2"])
y_train = train_df2["label"]

clf = SGDClassifier(loss="log_loss", random_state=12345)
clf.fit(X_train, y_train)

# Notice that you aren't fitting it again, we're just using the already trained count vectorizer to transform the test data here. Do not use fit_transform here!
X_test = vectorizer.transform(test_df2["text2"])

y_true = test_df2["label"]
y_pred = clf.predict(X_test)

print("SGD accuracy: %.3f" % (accuracy_score(y_true, y_pred)))



# (f) MLE

def mle_preprocess(s=""):
	s = s.lower()
	return reg_tokenizer.tokenize(s)

df2["text_ngram"] = df2["text"].map(mle_preprocess)


generate_nums = 1
classifications = []
ngrams = [2, 5]
ngrams = [] # skip

if len(ngrams) > 0:
	for n in ngrams:
		train_data, padded_sents = padded_everygram_pipeline(n, df2["text_ngram"])
		mle = MLE(n)
		mle.fit(train_data, padded_sents)
		document = ' '.join(mle.generate(15))
		X = vectorizer.transform([document])
		print("Classficiation for generated document with %d-gram: %s" % (n, clf.predict(X)))
		classification = []
		documents = []
		for i in range(generate_nums):
			document = ' '.join(mle.generate(20))
			documents.append(document)
		X = vectorizer.transform(documents)
		sentiment = clf.predict(X)
		classifications.append(sentiment)
	gram2_df = pd.DataFrame(data=classifications[0], columns=["label"])
	gram5_df = pd.DataFrame(data=classifications[1], columns=["label"])
	fig, axes = plt.subplots(1, 2, figsize=(13, 4))
	sns.barplot(data = gram2_df['label'].value_counts().to_frame(), x="count", y="label", ax=axes[0])
	sns.barplot(data = gram5_df['label'].value_counts().to_frame(), x="count", y="label", ax=axes[1])
	axes[0].set_title("2-gram sentiments distribution")
	axes[1].set_title("5-gram sentiments distribution")
	plt.show()


# (g) WHY IMBALANCED?


# 2-gram's distribution is more balanced.


# (h) CHRISTMAS EMOTIONS Word2Vec SKIP-GRAMS

vector_size = 25

w2v_model = Word2Vec(sentences=df2["text_ngram"], window=5, min_count=3, vector_size=vector_size, epochs=50, sg=1, workers=multiprocessing.cpu_count())


import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Code is taken from https://www.kaggle.com/code/ingledarshan/gensim-word2vec-tutorial#Training-the-model
def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, vector_size), dtype='f')
    word_labels = [word]
    color_list  = ['red']
    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = umap.UMAP().fit_transform(arrays)
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 },
                    )
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)
    plt.xlim(Y[:, 0].min()-0.5, Y[:, 0].max()+0.5)
    plt.ylim(Y[:, 1].min()-0.5, Y[:, 1].max()+0.5)
    plt.title('t-SNE visualization for {}'.format(word.title()))


word = "christmas"
tsnescatterplot(w2v_model, word, [i[0] for i in w2v_model.wv.most_similar(negative=[word])])
plt.show()

# Screenshot file: h0.png h1.png h2.png

# (i) EMBEDDINGS SIMILARITY



s = w2v_model.wv.most_similar(positive=[word], topn=3)
display(s)
print("Top 3 similarities: %.3f, %.3f, %.3f" % (s[0][1], s[1][1], s[2][1]))

ds = w2v_model.wv.most_similar(negative=[word], topn=3)
display(ds)
print("Top 3 dissimilarities: %.3f, %.3f, %.3f" % (ds[0][1], ds[1][1], ds[2][1]))


# It means two embeddings as vectors will point to close direction when they have a high cosine similarity


# (j) EXPLAIN SIMILAR/DISSIMILAR


similarity = w2v_model.wv.similarity(s[0][0], ds[0][0])
print(f"similarity between {s[0][0]} and {ds[0][0]}: {similarity:.3f}")

# Explanation: For the most similar word "christmas", its vector has the most similar direction to the vector for the word "christmas", whereas for the most dissimilar word "christmas" , its vector has the opposite direction to that of the vector for the word "christmas". So, these two vectors (the most similar and most dissimilar ones) point to opposite direction, and their cosine similarity will be small.

# (k) RELATED AND UNRELATED EMOTIONS TO CHRISTMAS

positive = w2v_model.wv.most_similar(topn=3, positive=["emotion", "christmas"])
negative = w2v_model.wv.most_similar(topn=3, negative=["emotion", "christmas"])
print("Related", positive)
print("Unrelated", negative)

# This method computes cosine similarity between a simple mean of the projection weight vectors of the given keys and the vectors for each key in the model.