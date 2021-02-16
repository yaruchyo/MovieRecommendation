import pandas as pd
import requests
import json

class GetPostersData:

    raw_data = pd.read_csv('app/static/datasets/netflix_titles.csv')  # read

    def get_posters_from_API(self,films):
        poster_array = []
        print(films)
        for index, film in films.iterrows():
            print(film['title'])
            search_str = film['title'].replace(' ', '+')
            year = film['years']
            url = 'http://www.omdbapi.com/?apikey=79d04ee6&t=%s&y=%s' % (search_str, year)
            print(url)
            response = requests.get(url)

            if response.text == '{"Response":"False","Error":"Movie not found!"}':
                url = 'http://www.omdbapi.com/?apikey=79d04ee6&t=%s' % (search_str)
                print('url2=', url)
                response = requests.get(url)

            json_file = json.loads(response.text)

            try:
                if json_file['Poster'] != "N/A":
                    poster_array.append(json_file['Poster'])
                else:
                    poster_array.append('static/img/keep-calm-poster-not-found.png')
            except:
                poster_array.append('static/img/keep-calm-poster-not-found.png')
        films['Poster'] = poster_array
        posts = films.to_dict('records')
        return posts

