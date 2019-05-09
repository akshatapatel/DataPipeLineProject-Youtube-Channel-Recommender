import indicoio
import numpy as np
import pandas as pd
indicoio.config.api_key = '7386b4fd42854ce9e37cdc049d9130ce'

class Sentiment():
	def get_sentiment(self,comments):
		sent = np.mean(indicoio.sentiment(comments))
		return sent

	def get_keywords(self,desc):
		kw = indicoio.keywords(desc, version=4)
		kwords = pd.DataFrame(kw, index=['weight']).T.reset_index().rename({'index':'word'}, axis='columns')
		return kwords