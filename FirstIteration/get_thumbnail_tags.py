#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
from urllib.request import urlopen
import urllib

from PIL import Image
from io import BytesIO


subscription_key = 'xxxxxx'
assert subscription_key

vision_base_url = "https://eastus.api.cognitive.microsoft.com/vision/v2.0/"
analyze_url = vision_base_url + "analyze"



# In[13]:

class ThumbnailTags():
    def get_tags(self,image_path):
        # image_data = open(image_path, "rb").read()

        try:
            image_data = urlopen(image_path).read()
            
        except urllib.error.HTTPError:
            return []
        
        headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                      'Content-Type': 'application/octet-stream'}
        params     = {'visualFeatures': 'Description'}
        response = requests.post(
            analyze_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()

      
        analysis = response.json()

        # image_caption = analysis["description"]["captions"][0]["text"].capitalize()
        image_tags = analysis['description']["tags"]

        return image_tags




