
import os
import pandas as pd
import googleapiclient.discovery


# In[2]:


from googleapiclient.errors import HttpError


# In[3]:


df = pd.DataFrame(columns=['video_id', 'title','channel_title','category_name','tags','views','subscribers','likes',
                           'dislikes','thumbnail_link','comments'])

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "xxxxxx"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)
    index=0
    for i in range(7):
        # print(i)
        if i==0:
            request = youtube.search().list(
                part="snippet",
                location="37.0902,264.2871",
                locationRadius="100mi",
                maxResults=50,
                q="surfing",
                type="video"
            )
        else:
            request = youtube.search().list(
                part="snippet",
                location="37.0902,264.2871",
                locationRadius="100mi",
                maxResults=50,
                pageToken=token,
                q="surfing",
                type="video")
                
        response = request.execute()
        token = response['nextPageToken']
        for it in response['items']:
            videoid = it['id']['videoId']
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=videoid
            )
            response1 = request.execute()
    
            for items in response1['items']:
                video_id = items['id']
                title = items['snippet']['title']
                thumbnail_link = items['snippet']['thumbnails']['default']['url']
                channel_title = items['snippet']['channelTitle']
                try:
                    tags = items['snippet']['tags']
                except KeyError:
                    tags = []
                category_id = items['snippet']['categoryId']
                views = items['statistics']['viewCount']
                try:
                    likes = items['statistics']['likeCount']
                except KeyError:
                    likes = ''
                try:
                    dislikes = items['statistics']['dislikeCount']
                except KeyError:
                    dislikes = ''
                channelId = items['snippet']['channelId']
                
                request = youtube.channels().list(
                    part="snippet,contentDetails,statistics",
                    id=channelId)
                response4 = request.execute()
                
                subscribers = response4['items'][0]['statistics']['subscriberCount']

                request = youtube.videoCategories().list(
                part="snippet",
                id=category_id)
                response2 = request.execute()

                category_name = response2['items'][0]['snippet']['title']

                request = youtube.commentThreads().list(
                    part="snippet,replies",
                    maxResults=100,
                    videoId=video_id,
                )
                try:
                    response3 = request.execute()
                    comments = []
                    for comment in response3['items']:
                        comments.append(comment['snippet']['topLevelComment']['snippet']['textOriginal'])
                except HttpError:
                    comments = []

                df.loc[index] = [video_id] + [title] + [channel_title] + [category_name] + [tags] + [views] + [subscribers] + [likes] + [dislikes] + [thumbnail_link] + [comments]

                index = index + 1

if __name__ == "__main__":
    res = main()


df.to_csv('YoutubeData.csv',index=False)





