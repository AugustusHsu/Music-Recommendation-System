'''
A Example to call the Search_Tag:
Tag_Dic = {}
artist_name = 'x japan'
track_name = 'art of life'
Tag_Dic = Search_Tag(Tag_Dic, artist_name, track_name)

print(Tag_Dic)

Output:

{'Masterpiece': [['x japan', 'art of life']], 'J-rock': [['x japan', 'art of life']], 'epic': [['x japan', 'art of life']]}

'''

import requests
import time
import json

API_KEY = 'c3ac6bf07c1c424328104056e80e08ee'
ALL_TAG_URL = 'https://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_unique_tags.txt'
Search_URL = 'http://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={}&track={}&api_key={}&format=json'
top_k = 3

data_dir = 'data/'


def Not_Enough_Tags(Invalid, artist_name, track_name):
    #print('Error! Not enough three tags at artist name: {} track_name: {}'.format(artist_name, track_name))
    Invalid.append([artist_name, track_name])
    #return Invalid

def Empty_Tags(Invalid, artist_name, track_name):
    #print('Error! ' + str(reqs) + ' at artist name: {} track_name: {}'.format(artist_name, track_name))
    Invalid.append([artist_name, track_name])
    #return Invalid
        
def _Search_Tag(Unique_List, Tag_List, Invalid, artist_name, track_name):
    artist_str = artist_name.replace(' ', '+')
    track_str = track_name.replace(' ', '+')
    
    try:
        reqs = requests.get(Search_URL.format(artist_str, track_str, API_KEY))
        reqsjson = json.loads(reqs.text)

        #time.sleep(0.001)
        if len(reqsjson['toptags']['tag']) > top_k:
            Unique_List.append([artist_name, track_name])
            for idx in range(top_k):
                name = reqsjson['toptags']['tag'][idx]['name']
                #print(name)
                if idx == 0:
                    Tag_List.append([])
                    Tag_List[len(Tag_List)-1].append(name)
                else:
                    Tag_List[len(Tag_List)-1].append(name)
        else:
            Invalid.append([artist_name, track_name])
            #Not_Enough_Tags(artist_name, track_name)
            

    except:
        Invalid.append([artist_name, track_name])
        #Empty_Tags(reqs, artist_name, track_name)
    #return Unique_List, Tag_List, Invalid

def Search_Tag(Unique_List, Tag_List, Invalid, artist_name, track_name):
    _Search_Tag(Unique_List, Tag_List, Invalid, artist_name, track_name)
    
    return Unique_List, Tag_List, Invalid

