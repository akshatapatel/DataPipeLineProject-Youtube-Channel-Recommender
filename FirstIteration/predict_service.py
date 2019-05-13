
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop, PeriodicCallback
import MySQLdb.connections
import pandas as pd
import numpy as np
import json
from gensim import models
import pickle
import indico
import get_thumbnail_tags
import ast


# In[77]:

class LoadModel():
    with open('one_mill_trained_model.pkl', 'rb') as fp:
        w2v = pickle.load(fp)

class GetPredData():
    def get_data(self):
        mydb = MySQLdb.connect(host="127.0.0.1", user="root", passwd="xxxx", db = "DataPipeline")
        youtube_data = pd.read_sql("""SELECT * from YoutubeData""",con=mydb)
        # youtube_data = pd.read_csv('YoutubeData_updated.csv')
        return youtube_data




class PredictHandler(GetPredData):
    def get_text(self):
        youtube_data= self.get_data()
        return youtube_data
    
    def get_similarity_score(self,user_pc, user_vc, user_desc_df, user_weight, yt_cn, yt_t, yt_thumb_kw): #add thumbnail once done

        load_model = LoadModel()

        #convert category name to a list of words - if the word is in the word2vec vocab
        list_cn = [word for word in yt_cn.split() if word in load_model.w2v.vocab]
        list_kw = [word for word in yt_thumb_kw if word in load_model.w2v.vocab]
       

        #convert product category to a list of words - if the word is in the word2vec vocab
        list_pc = [word for word in user_pc.split() if word in load_model.w2v.vocab]
        

        if list_pc and list_cn: #if lists are both non-empty, calculate their similarity
            pc_cn_similarity = load_model.w2v.n_similarity(list_pc, list_cn)
        else:
            pc_cn_similarity = 0.0


        if list_pc and list_kw: #if lists are both non-empty, calculate their similarity
            pc_kw_similarity = load_model.w2v.n_similarity(list_pc, list_kw)
        else:
            pc_kw_similarity = 0.0
            
        pc_tag_similarity = []
        
        for tag in yt_t: #for each tag, repeat the same procedure as above
            list_tag = [word for word in tag.split() if word in load_model.w2v.vocab]
            if list_pc and list_tag: #if lists are both non-empty, calculate their similarity
                pc_tag_similarity.append(load_model.w2v.n_similarity(list_pc, list_tag))
            else:
                pc_tag_similarity.append(0.0)

        pc_tag_similarity = np.array(pc_tag_similarity)
        pc_tag_similarity_avg = np.average(pc_tag_similarity[np.flip(np.argsort(pc_tag_similarity))[:5]]) #get the average score of the top 5 most similar tags

        

        #repeat the same for video category
        list_vc = [word for word in user_vc.split() if word in load_model.w2v.vocab]
        
        if list_vc and list_cn:
            vc_cn_similarity = load_model.w2v.n_similarity(list_vc, list_cn)
        else:
            vc_cn_similarity = 0.0


        if list_vc and list_kw: #if lists are both non-empty, calculate their similarity
            vc_kw_similarity = load_model.w2v.n_similarity(list_vc, list_kw)
        else:
            vc_kw_similarity = 0.0
            

        vc_tag_similarity = []
        
        for tag in yt_t:
            list_tag = [word for word in tag.split() if word in load_model.w2v.vocab]
            if list_vc and list_tag:
                vc_tag_similarity.append(load_model.w2v.n_similarity(list_vc, list_tag))
            else:
                vc_tag_similarity.append(0.0)
                
        vc_tag_similarity = np.array(vc_tag_similarity)
        vc_tag_similarity_avg = np.average(vc_tag_similarity[np.flip(np.argsort(vc_tag_similarity))[:5]])
        
        #repeat the same for product description


        #convert description to a list of words - if the word is in the word2vec vocab
        list_desc_all = [word for word in user_desc_df.word if word in load_model.w2v.vocab]
        list_desc_weights = [user_desc_df.weight[ind] for ind,word in enumerate(user_desc_df.word) if word in load_model.w2v.vocab] #get corresponding weights

        desc_cn_similarity_each = []

        #if both lists are not empty, get similarity of every word in the description to the category name
        if list_desc_all and list_cn:
            for keywrd in list_desc_all:
                desc_cn_similarity_each.append(load_model.w2v.n_similarity([keywrd], list_cn))

            desc_cn_similarity = np.average((desc_cn_similarity_each), weights = list_desc_weights) #average out all similarities based on the keyword wieghts

        else:
            desc_cn_similarity = 0.0


        desc_kw_similarity_each = []

        if list_desc_all and list_kw:
            for keywrd in list_desc_all:
                desc_kw_similarity_each.append(load_model.w2v.n_similarity([keywrd], list_kw))

            desc_kw_similarity = np.average((desc_kw_similarity_each), weights = list_desc_weights) #average out all similarities based on the keyword wieghts

        else:
            desc_kw_similarity = 0.0



        desc_tag_similarity = []

        
        for tag in yt_t: #for each tag
            list_tag = [word for word in tag.split() if word in load_model.w2v.vocab]
            #if both lists are not empty, get similarity of every word in the description to the tag
            desc_tag_similarity_each = []
            if list_desc_all and list_tag:
                for keywrd in list_desc_all:
                    desc_tag_similarity_each.append(load_model.w2v.n_similarity([keywrd], list_tag))


                desc_tag_similarity.append(np.average((desc_tag_similarity_each), weights = list_desc_weights)) #average out all similarities based on the keyword wieghts

            else:
                desc_tag_similarity.append(0.0)

        desc_tag_similarity = np.array(desc_tag_similarity)
        desc_tag_similarity_avg = np.average(desc_tag_similarity[np.flip(np.argsort(desc_tag_similarity))[:5]])  #get the average score of the top 5 most similar tags


        #get average similarity value for each - product category, video category and product description
        pc_avg = np.average((pc_cn_similarity, pc_tag_similarity_avg, pc_kw_similarity))
        vc_avg = np.average((vc_cn_similarity, vc_tag_similarity_avg, vc_kw_similarity))
        desc_avg = np.average((desc_cn_similarity, desc_tag_similarity_avg, desc_kw_similarity))
        

        similarity_score = (user_weight[0]*pc_avg) + (user_weight[1]*vc_avg) + (user_weight[1]*desc_avg) #perform weighted average of the above, based on weights provided by user

        return similarity_score
    
    def predict(self, product_category, video_category, product_description, weightage):
        user_pc = product_category
        user_vc = video_category
        user_desc = product_description
        user_weight = [float(wght) for wght in weightage.split(',')]

        youtube_data = self.get_text()
        yt_cn_all = youtube_data['category_name']
        yt_t_all = youtube_data['tags']
        yt_thumbnail_all = youtube_data['thumbnail_tags']

        similarity_score_all = []

        sent = indico.TextAnalysis()
        user_desc_df = sent.get_keywords(user_desc)

        for i in range(yt_cn_all.shape[0]):

            similarity_score_all.append(self.get_similarity_score(user_pc, user_vc, user_desc_df, user_weight, yt_cn_all.iloc[i], yt_t_all.iloc[i], yt_thumbnail_all.iloc[i])) 
            

        return similarity_score_all, youtube_data



class RecommendationHandler(PredictHandler):
    def recommend_channel(self, product_category, video_category, product_description, weightage): #ADD COMMENTS SENTIMENT FEATURE
        similarity_score_all, youtube_data = np.array(self.predict(product_category, video_category, product_description, weightage))
        

        youtube_data['similarity_score'] = similarity_score_all
        df_category = youtube_data.groupby('channel_title', as_index=False).mean().sort_values(by=['similarity_score'], ascending=False)
        df_category['diff_likes_dislikes'] = df_category['likes'] - df_category['dislikes']
        top_5_channels = df_category.iloc[:10].sort_values(by=['diff_likes_dislikes'], ascending=False).iloc[:5]
        # print(top_5_channels[['channel_title','likes','dislikes','views','subscribers','sentiment_score']])

        results = top_5_channels[['channel_title','likes','dislikes','views','subscribers','sentiment_score']].iloc[:5]
        list_of_channels = results['channel_title'].values.tolist()
        list_of_likes = results['likes'].values.tolist()
        list_of_dislikes = results['dislikes'].values.tolist()
        list_of_views = results['views'].values.tolist()
        list_of_subscribers = results['subscribers'].values.tolist()
        list_of_sentiment_score = results['sentiment_score'].values.tolist()
        full_list = [str(list_of_channels),str(list_of_likes),str(list_of_dislikes),str(list_of_views),str(list_of_subscribers),str(list_of_sentiment_score)]

        return "-".join(full_list)

       