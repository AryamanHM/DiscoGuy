from flask import Flask, request, render_template
import os

# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)

import pandas as pd
import numpy as np
from numpy import int64

import requests
import IPython.display as Disp

import sklearn
from sklearn.decomposition import TruncatedSVD





@app.route("/")
def hello():
	return TEMPLATE_DIR



print("building recommendation engine")
print("reading songs data ")
songs_metadata_file = 'C:/Users/aryam/Desktop/ML/song_data.csv'
songs_df =  pd.read_csv(songs_metadata_file)
print("done")



print("reading user data about songs")
triplets_file = 'C:/Users/aryam/Desktop/ML/10000.txt'
songs_to_user_df = pd.read_table(triplets_file,header=None)
songs_to_user_df.columns = ['user_id', 'song_id', 'listen_count']
print("done")



print("merging songs and user data")
combined_songs_df = pd.merge(songs_to_user_df, songs_df, on='song_id')
print("done")

print("creating pivot table")
ct_df = combined_songs_df.pivot_table(values='listen_count', index='user_id', columns='title', fill_value=0)
print("done")


print("applying SVD")
X = ct_df.values.T
SVD  = TruncatedSVD(n_components=20, random_state=17)
result_matrix = SVD.fit_transform(X)
print("done")

print("crerating pearson corff matrix")
corr_mat = np.corrcoef(result_matrix)
corr_mat.shape
print("done")

song_names = ct_df.columns
song_list = list(song_names)
print("done building recommendation engine")
print("ready for recommendation engine")
#hunger_game_index = book_list.index('The Hunger Games')
#corr_hunger_games = corr_mat[hunger_game_index]
#list(book_names[(corr_hunger_games<1.0) & (corr_hunger_games>0.8)])


def getRecommendations(songName):
	query_index = song_list.index(songName)

	corr_similar_songs = corr_mat[query_index]
	recList = list(song_names[(corr_similar_songs<1.0) & (corr_similar_songs>0.98)])
	max=5
	if(len(recList)<5):
		max=len(recList)
	return songs_df[songs_df.title.isin(recList)]



@app.route("/rec",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		print(str(request.form.get('query')))
		query = request.form.get('query')
		#print("the book name is " + query)
		recommendations = getRecommendations(query)
		#print(query)
		return render_template('rec.html', query=query, recommendations=recommendations.to_html())
	else:
		return render_template('rec.html', query="" ,recommendations="<<unknown>>")
	

if __name__ == "__main__":
    app.run(debug=True)


