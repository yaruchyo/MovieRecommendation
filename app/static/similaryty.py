import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


X_PCA = np.load('app/static/npy/pca.npy')
KNN_PCA = NearestNeighbors(n_neighbors=13 , algorithm='kd_tree', metric= 'cityblock')
KNN_PCA.fit(X_PCA)


class MovieSimilarity:
    raw_data = pd.read_csv('app/static/datasets/netflix_titles.csv')
    cosine_sim = np.load('app/static/npy/pca.npy') #has to be cosine_sim.npy

    indices = pd.Series(raw_data['title'])

    def init_filename(self , str):
        print(str)

        self.splited_array = str.rsplit("(")

        self.movie_title = (self.splited_array[0])[:-1]
        self.year = (self.splited_array[1]).replace(')',"")


    def recommendations(self, cosine_sim=cosine_sim):
        recommended_movies = []
        recommended_year = []

        # gettin the index of the movie that matches the title

        filt = self.raw_data['title'].str.find(self.movie_title) >= 0
        data = self.raw_data.loc[filt]
        filt  = data['release_year'] == int(self.year)
        idx = (data.loc[filt]['title'].index[0])

        #idx = self.indices[self.indices == self.movie_title].index[0]
        # creating a Series with the similarity scores in descending order
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=True)
        # print(score_series)
        # getting the indexes of the 10 most similar movies
        top_10_indexes = list(score_series.iloc[1:13].index)
        # populating the list with the titles of the best 10 matching movies
        for i in top_10_indexes:
            recommended_movies.append(list(self.raw_data['title'])[i])
            recommended_year.append(list(self.raw_data['release_year'])[i])
        d = {'title': recommended_movies, 'years': recommended_year}
        return pd.DataFrame(data=d)

    def recommendations_knn_pca(self):
        print('read PCA')
        filt = self.raw_data['title'].str.find(self.movie_title) >= 0
        data = self.raw_data.loc[filt]
        filt = data['release_year'] == int(self.year)
        idx = (data.loc[filt]['title'].index[0])

        coordinates = X_PCA[idx]
        # print(neigh.kneighbors([coordinates]))

        top_10_knn = KNN_PCA.kneighbors([coordinates])
        # getting the indexes of the 10 most similar movies
        top_10_indexes = list(top_10_knn[1][0])
        top_10_indexes.pop(0)

        recommended_movies = []
        recommended_year = []

        # populating the list with the titles of the best 10 matching movies
        for i in top_10_indexes:
            recommended_movies.append(list(self.raw_data['title'])[i])
            recommended_year.append(list(self.raw_data['release_year'])[i])
        d = {'title': recommended_movies, 'years': recommended_year}
        return  pd.DataFrame(data=d)


    def random_from_pd(self):
        random_data = self.raw_data.sample(12)

        data_to_return =  random_data[['title','release_year']]
        data_to_return = data_to_return.rename(columns={'release_year':'years'})
        return data_to_return



#movie_recomender = MovieSimilarity()
#movie_recomender.init_filename('The Irishman (2015)')
#print(movie_recomender.recommendations())
#movie_recomender.random_from_pd()