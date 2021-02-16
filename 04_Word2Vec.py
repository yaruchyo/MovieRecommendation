# import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk.data
import gensim
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
# from gensim.models import KeyedVectors

# Read data from CSV
data = pd.read_csv('/home/elisabeth/Dokumente/EMLP/archive/netflix_titles.csv')

# Preprocessing: change review to a list of words (because Word2Vec needs Input in that form)
nltk.download('stopwords')


def description_to_wordlist(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words, removing stop words; returns a list of words;

    # Remove non-letters
    description_text = re.sub("[^a-zA-Z]", " ", review)

    # Convert words to lower case and split them
    words = description_text.lower().split()

    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    # Return a list of words
    return words


# create a list of all descriptions as list of words (list of lists)
corpus = []  # Initialize an empty list of sentences

for description in data['description']:
    description_list = description_to_wordlist(description)
    corpus.append(description_list)

print(corpus[6233])

# Training our corpus with Google Pretrained Model
embedding_file = '/home/elisabeth/Dokumente/EMLP/GoogleNews-vectors-negative300.bin.gz'
google_word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)

google_model = Word2Vec(size=300, window=5, min_count=2, workers=-1)
google_model.build_vocab(corpus)
google_model.intersect_word2vec_format(embedding_file, lockf=1.0, binary=True)
google_model.train(corpus, total_examples=google_model.corpus_count, epochs=5)

# Creating a list for storing the vectors (description into vectors)
word_embeddings = []


# Generate the average word2vec for each movie description
def vectors():

    global word_embeddings

    # Reading each description
    for line in corpus:
        avgword2vec = None
        count = 0
        for word in line:
            if word in google_model.wv.vocab:
                count += 1
                if avgword2vec is None:
                    avgword2vec = google_model[word]
                else:
                    avgword2vec = avgword2vec + google_model[word]

        if avgword2vec is not None:
            avgword2vec = avgword2vec / count

            word_embeddings.append(avgword2vec)
            # print(word_embeddings)


# Dimensional reduction with PCA

# Standardization of Vectors in word_embeddings
# word_embeddings_std = StandardScaler().fit_transform(word_embeddings)

# Setting the favored explained variance
# explained_variance = 0.95

# Apply PCA with favored variance to standardized word_embeddings and print
# pca = PCA(explained_variance)
# word_embeddings_pca = pca.fit(word_embeddings_std)
# print('We have now'+pca.n_components_+'dimensions')


# Recommending the Top 5 similar books
def recommendations(title):

    # Calling the function vectors
    vectors()

    # finding cosine similarity for the vectors
    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)

    recommended_movies = []

    # Reverse mapping of the index
    indices = pd.Series(data['title'])

    # getting the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    # getting the indices of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:10].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(data['title'])[i])

    return recommended_movies


print(recommendations('Avengers: Infinity War'))
