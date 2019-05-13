import pandas as pd
import numpy as np
import indico
import get_thumbnail_tags
import ast

class PreCompute():
	def calc_sentiment(self,youtube_data):
		comments = youtube_data['comments']
		sent = indico.TextAnalysis()
		sentiment_score = []
		
		for comment in comments:
			com = ast.literal_eval(comment)
			sentiment_score.append(sent.get_sentiment(com))
			
		youtube_data['sentiment_score'] = sentiment_score

		return youtube_data

	def calc_thumbnail_tags(self,youtube_data):
		thumb_tag = get_thumbnail_tags.ThumbnailTags()
		thumbnails = youtube_data['thumbnail_link']

		thumbnail_tags = []
		for thumbnail in thumbnails:
			thumbnail_tags.append(thumb_tag.get_tags(thumbnail))
			

		youtube_data['thumbnail_tags'] = thumbnail_tags

		return youtube_data


precomp = PreCompute()
df = pd.read_csv('YoutubeData.csv')
df = precomp.calc_sentiment(df)
df = precomp.calc_thumbnail_tags(df)

df.to_csv('YoutubeData_updated.csv')