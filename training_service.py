import flask
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

from sklearn.linear_model import LinearRegression
#from database import get_data
import pickle
import pandas as pd
from gensim import models

def train_model():
    # using the pre-trained word2vec model for the MVP
    w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True,limit=1000000)
    return w

class ModelHandler(Resource):
    #presumably, this should just live in a shared library
#     def __init__(self, init_model=False):
#         if init_model:
#             self.model = self.load_model()

#     def load_model(self):
#         self.model = pd.read_pickle("./1_mill_trained_model.pkl",'zip')
#         return self.model

    @app.route("/", methods=['GET'])
    def save_model():
        w = train_model()
        word_list = []
        vector_list = []

        for word in w.vocab:
            word_list.append(word)
            vector_list.append(w[word])
        
        word2vec_df = pd.DataFrame({'word':word_list,'vector':vector_list})
        word2vec_df.to_pickle("./1_mill_trained_model.pkl",'zip')
        return "Welcome to our Youtube Influencer Recommendaion System"

api.add_resource(ModelHandler, '/training_service')

if __name__ == '__main__':
    app.debug = True
    app.run(debug=False)
    