import indicoio
import numpy as np
import pandas as pd
indicoio.config.api_key = 'c43834b9748e32e0826fb508d52663b9'

class TextAnalysis():
	def get_sentiment(self,comments):
		if comments:
			sent = np.mean(indicoio.sentiment(comments))
		else:
			sent=0.0
		return sent

	def get_keywords(self,desc):
		kw = indicoio.keywords(desc, version=4)
		kwords = pd.DataFrame(kw, index=['weight']).T.reset_index().rename({'index':'word'}, axis='columns')
		return kwords