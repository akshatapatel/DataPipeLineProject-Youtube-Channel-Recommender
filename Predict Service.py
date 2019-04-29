
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop, PeriodicCallback
import MySQLdb.connections
import pandas as pd
import numpy as np
import json
from gensim import models
import pickle


# In[77]:

class LoadModel():
    with open('one_mill_trained_model.pkl', 'rb') as fp:
        w2v = pickle.load(fp)

class GetPredData(): #params - RequestHandler, MySQLdb.connections.Connection
    user_data = pd.DataFrame({'Product Category': 'makeup','Video Category': 'style'}, index=[0])
    def get_data(self):
        mydb = MySQLdb.connect(host="127.0.0.1", user="root", passwd="DataPipeline", db = "DataPipeline")
        youtube_data = pd.read_sql("""SELECT * from USVideos""",con=mydb)
        return self.user_data,youtube_data


# In[78]:


class PredictHandler(GetPredData):
    def get_text(self):
        user_data, youtube_data= self.get_data()
        user_pc = self.user_data['Product Category'][0]
        user_vc = self.user_data['Video Category'][0]
        return user_pc, user_vc, youtube_data
    
    def get_similarity_score(self,user_pc, user_vc, yt_cn, yt_t):
        
        load_model = LoadModel()
        list_pc = [word for word in user_pc.split() if word in load_model.w2v.vocab]
        list_cn = [word for word in yt_cn.split() if word in load_model.w2v.vocab]
        
        if list_pc and list_cn:
            pc_cn_similarity = load_model.w2v.n_similarity(list_pc, list_cn)
        else:
            pc_cn_similarity = 0.0
            
        pc_tag_similarity = []
        for tag in yt_t.split('|'):
            list_tag = [word for word in tag.split() if word in load_model.w2v.vocab]
            if list_pc and list_tag:
                pc_tag_similarity.append(load_model.w2v.n_similarity(list_pc, list_tag))
            else:
                pc_tag_similarity.append(0.0)
        pc_tag_similarity = np.array(pc_tag_similarity)
        pc_tag_similarity_avg = np.average(pc_tag_similarity[np.flip(np.argsort(pc_tag_similarity))[:5]])

        
        
        list_vc = [word for word in user_vc.split() if word in load_model.w2v.vocab]
        
        if list_vc and list_cn:
            vc_cn_similarity = load_model.w2v.n_similarity(list_vc, list_cn)
        else:
            vc_cn_similarity = 0.0
            
        vc_tag_similarity = []
        for tag in yt_t.split('|'):
            list_tag = [word for word in tag.split() if word in load_model.w2v.vocab]
            if list_vc and list_tag:
                vc_tag_similarity.append(load_model.w2v.n_similarity(list_vc, list_tag))
            else:
                vc_tag_similarity.append(0.0)
                
        vc_tag_similarity = np.array(vc_tag_similarity)
        vc_tag_similarity_avg = np.average(vc_tag_similarity[np.flip(np.argsort(vc_tag_similarity))[:5]])
        

        pc_avg = np.average((pc_cn_similarity, pc_tag_similarity_avg))
        vc_avg = np.average((vc_cn_similarity, vc_tag_similarity_avg))

        similarity_score = ((0.5*pc_avg )+ (0.5*vc_avg))/2
        return similarity_score
    
    def predict(self):
        user_pc, user_vc, youtube_data = self.get_text()
        yt_cn_all = youtube_data['category_name'].str[:-1]
        yt_t_all = youtube_data['tags']
        similarity_score_all = []
        for i in range(yt_cn_all.shape[0]):
            similarity_score_all.append(self.get_similarity_score(user_pc, user_vc, yt_cn_all.iloc[i], yt_t_all.iloc[i]))

        return similarity_score_all, youtube_data


# In[84]:


class RecommendationHandler(PredictHandler):
    def recommend_channel(self):
        similarity_score_all, youtube_data = np.array(self.predict())
        youtube_data['similarity_score'] = similarity_score_all
        df_category = youtube_data.groupby('channel_title', as_index=False).mean().sort_values(by=['similarity_score'], ascending=False)
        df_category['diff_likes_dislikes'] = df_category['likes'] - df_category['dislikes']
        top_5_channels = df_category.iloc[:10].sort_values(by=['diff_likes_dislikes'], ascending=False).iloc[:5]
        return top_5_channels[['channel_title','likes','dislikes']]


# In[85]:


rh = RecommendationHandler()
recommendations = rh.recommend_channel()


# In[86]:


print(recommendations)


# In[ ]:




