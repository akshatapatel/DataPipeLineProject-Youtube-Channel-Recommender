# coding: utf-8

# In[1]:


import flask
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


# In[2]:


#from database import get_data
import pickle
import pandas as pd
from gensim import models


# In[3]:


def train_model():
    # using the pre-trained word2vec model for the MVP
    w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True,limit=1000000)
    return w


# In[4]:


class ModelHandler(Resource):

    @app.route("/training_service", methods=['GET'])
    def save_model():
        w = train_model()
        with open('one_mill_trained_model.pkl', 'wb') as fp:
            pickle.dump(w, fp)
            
        return "Welcome to our Youtube Influencer Recommendaion System"


# In[5]:


api.add_resource(ModelHandler, '/training_service')


# In[6]:


if __name__ == '__main__':
    app.debug = True
    app.run(debug=False)
