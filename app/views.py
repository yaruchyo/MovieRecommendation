from app import app
from flask import render_template, request
from app.static.poster_image import GetPostersData
from app.static.similaryty import MovieSimilarity


poster_data = GetPostersData()
movie_recommender = MovieSimilarity()

@app.route("/")
def main():
    title = 'Random films'
    random_films = movie_recommender.random_from_pd()
    posts = poster_data.get_posters_from_API(random_films)
    return render_template("home.html", post = posts, title = title,js_name = "static/js/myScript.js")

@app.route("/post", methods=['GET', 'POST'] )
def post():
    if len(request.values) >1:
        film = ''
        for i in request.values.to_dict():
            if i =='film':
                film = request.values.to_dict()[i]
            else:
                film = film +"&"+ i
    else:
        film = (request.values.to_dict())['film']

    movie_recommender.init_filename(film)
    #recommended_movies = movie_recommender.recommendations()
    recommended_movies = movie_recommender.recommendations_knn_pca()
    #posts = poster_data.random_img_from_pd()
    posts = poster_data.get_posters_from_API(recommended_movies)
    title = 'Similar to: '+film

    return render_template("home.html", post = posts, title = title, js_name = "static/js/myScript.js")

@app.errorhandler(404)
def page_not_found(error):
   return render_template('404.html', title = '404'), 404