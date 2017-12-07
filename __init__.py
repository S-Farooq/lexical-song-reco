from time import time
import numpy as np

#import matplotlib
#import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

#import matplotlib

#mpl.rcParams['figure.figsize'] = (20,10)

from sklearn.metrics.pairwise import euclidean_distances
# from mpl_toolkits.mplot3d import Axes3D

import re
import requests  
from bs4 import BeautifulSoup
import nltk

nltk.data.path.append("/home/shahamfarooq/nltk_data/")
nltk.data.path.append("/home/shahamfarooq/miniconda2/lib/nltk_data/")
nltk.data.path.append("/home/shahamfarooq/miniconda2/nltk_data/")
from nltk.tag.perceptron import PerceptronTagger

import cPickle
from io import BytesIO  

np.random.seed(42)

def get_normalized_and_split_data(all_data,x_names,split=0.2,by_artist=False):

    if by_artist:
        artists = all_data['artist'].unique()
        X_train = all_data[all_data['artist'].isin(artists[:-2])][x_names].values
        y_train = list(all_data[all_data['artist'].isin(artists[:-2])]['labels'])

        X_test = all_data[all_data['artist'].isin(artists[-2:])][x_names].values
        y_test = list(all_data[all_data['artist'].isin(artists[-2:])]['labels'])
    else:
        y = list(all_data['labels'])
        X = all_data[x_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=3)
    
    scaler = StandardScaler()
    X_train =scaler.fit_transform(X_train)
    if len(y_test)>0:
        X_test = scaler.transform(X_test)

    to_drop=[]
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if abs(X_train[i][j])>5:
                to_drop.append(i)

    to_drop=list(set(to_drop))

    X_train = np.delete(X_train, to_drop, axis=0)
    y_train = np.delete(y_train, to_drop, axis=0)

    n_samples, n_features = X_train.shape

    return X_train, X_test, y_train, y_test, scaler


def get_fitted_clusters(X_train,X_test,y_train,y_test,x_names,user_song_name="",user_scaled_data=[],n_clusters = 5):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)

    cluster_df = pd.DataFrame(kmeans.transform(X_train))
    cluster_df['labels'] = y_train
    cluster_df['pred_labels'] = kmeans.predict(X_train)
    cluster_df['test'] = 0

    if len(y_test)>0:
        cluster_dft = pd.DataFrame(kmeans.transform(X_test))
        cluster_dft['labels'] = y_test
        cluster_dft['pred_labels'] = kmeans.predict(X_test)
        cluster_dft['test'] = 1

    if len(user_scaled_data)>0:
        cluster_dft = pd.DataFrame(kmeans.transform(user_scaled_data))
        cluster_dft['labels'] = user_song_name
        cluster_dft['pred_labels'] = kmeans.predict(user_scaled_data)
        cluster_dft['test'] = 2

    cluster_df = cluster_df.append(cluster_dft)
    num=7

    check_c = cluster_df[cluster_df['test']==2]['pred_labels'].unique()
    for c in check_c:
        print "YOUR SONG FALLS UNDER CLUSTER#", c
        tmp = cluster_df[cluster_df['pred_labels']==c]
        tmp_train = tmp[tmp['test']==0].sort_values(c,ascending=True)
        msg= "<br>--TOP {num} Songs from my playlist lexically similar to yours (clustering):<br>".format(num=num)
        MyPrettyPrinter().pprint(list(tmp_train['labels'])[:num])
        to_return = msg+str(list([str(x)+"<br>" for x in tmp_train['labels'][:num]]))+"<br>"
        
    df_centroid = pd.DataFrame(kmeans.cluster_centers_)
    
    print"\n PLOT OF CLUSTER QUALITIES vs YOUR SONG:"
    df_centroid.columns = x_names
    if len(user_scaled_data)>0:
        df_tmp = pd.DataFrame(user_scaled_data)
        df_tmp.columns = x_names
        df_tmp.index = [user_song_name]
        df_centroid = df_centroid.append(df_tmp)
    df_centroid = df_centroid.transpose()

    #ax1 = df_centroid.plot(x_compat=True, lw=3)
    #fig = plt.xticks(range(len(df_centroid.index)), x_names, rotation=90)
    #ax1.set_ylim(-4,4)
    #plt.grid(True)
    #plt.savefig('tmp_plot.png')

    return to_return.encode(encoding='UTF-8',errors='ignore')



def get_euc_dist(set1,set2,set1_y,set2_y,n_top=10):
    ed_df = pd.DataFrame()
    ed = euclidean_distances(set1, set2)
    
    for n in range(len(set1_y)):
        df=pd.DataFrame()
        df['distance']=ed[n,:]
        df['to']=set2_y
        df['from']=set1_y[n]
        df['to_ind']=range(len(set2_y))
        df['from_ind']=n
        ed_df = ed_df.append(df)

    cols = ['from','to','from_ind','to_ind','distance']
    ed_df_p = ed_df[cols]
    ed_df_p = ed_df_p.sort_values(['from_ind','distance'],ascending=True)

    ed_df_p = ed_df_p.groupby('from').head(n_top)
    
    cols = ['from','to','distance']
    ed_df = ed_df[cols]
    ed_df = ed_df.sort_values(['from','distance'],ascending=True)
    
    ed_df_top = ed_df.groupby('from').head(n_top)
    ed_df_top['rel_conf'] = ed_df_top['distance']/ed_df_top['distance'].max()
    ed_df_top['rel_conf'] = ed_df_top['rel_conf']-ed_df_top['rel_conf'].min()
    ed_df_top['rel_conf'] = (1.0-ed_df_top['rel_conf'])*100

    ed_df_top = ed_df_top.rename(columns={'to': 'My_Songs', 'rel_conf': 'Relative_Confidence'})
    msg = "<br>Closest (lexically) songs from my playlist to your selected song:<br>"
    return ed_df_top[['My_Songs','distance']].sort_values(['distance'],ascending=True).to_html(index=False, header=False)

def search_musix_track(search_term):
    p = re.compile('\/lyrics\/*')

    webpage = "https://www.musixmatch.com"
    url = 'https://www.musixmatch.com/search/'
    useragent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'

    searchlink = url + search_term + "/tracks"
    try:
        r  = requests.get(searchlink, headers={'User-Agent': useragent})
    except Exception as e:
        print "No track found...something went wrong.."
    	print str(e)
    	return ""
    
    soup = BeautifulSoup(r.content, "lxml")
    gdata = soup.find_all('a',{'class':'title'})
    for s in range(len(gdata)):
        print s, "-",str(gdata[s].get('href')) 
    id = 0
    
    slink = str(gdata[id].get('href'))
    if p.search(slink):
        song_lyrics = []
        song_page = webpage+slink
        print song_page
        try:
            r  = requests.get(song_page, headers={'User-Agent': useragent})
        except:
            print "cant access lyric page for {song_page},.".format(song_page=song_page)
            return ""
        soup = BeautifulSoup(r.content, "lxml")
        song_name = soup.find_all('h1',{'class':'mxm-track-title__track'})
        song_name = song_name[0].text
        song_name = song_name[6:]
        print song_name
        
        lyrics = soup.find_all('p',{'class':'mxm-lyrics__content'})
        if len(lyrics)==0:
            print "No lyrics found for {song}".format(song=song_page)
        for l in lyrics:
            song_lyrics.append(l.text)

        song_lyrics  = "\n".join(song_lyrics)
        song_lyrics = song_lyrics.replace("\n\n","\n")
        return song_lyrics


def ie_preprocess(document):
    
    tagger = PerceptronTagger() 
    tagged = []
    sentences = document.split("\n")
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for sent in sentences:
        tagged_tokens = tagger.tag(sent)
        tagged.append(tagged_tokens)
    
    return tagged

def tokenize_song(lyrics):
    tokenized_lyrics = ie_preprocess(lyrics)
    return tokenized_lyrics




from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist
from collections import Counter
import pandas as pd

sid = SentimentIntensityAnalyzer()
punc = re.compile('\p')

def unique_words(text):
    return len(set(text))*1.0

def total_words(text):
    return len(text)*1.0

def lexical_diversity(text):
    return unique_words(text) / total_words(text)

def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return len(unusual)

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / total_words(text)

def avg_len_words(text):
    avg_len =0.0
    for t in text:
        avg_len+=len(t)
    return avg_len/len(text)

def get_pos_stats(tagged):
    stats =[]
    names =[]
    counts = Counter(tag for word,tag in tagged)
    total = sum(counts.values())
    counts_perc = dict((word, float(count)/total) for word,count in counts.items())
#     print counts_perc
    for t in ['NN','PR','RB','RBR','RBS','UH','VB','JJ','JJR','JJS','EX']:
        names.append(t)
        pa = 0.0
        for key in counts_perc.keys():
            if key.startswith(t):
                pa+=counts_perc[key]
        stats.append(pa)
#                 try:
#                     stats.append(counts_perc[t])
#                 except:
#                     stats.append(0.0)
    
    return stats, names

def get_word_stats(input_text):
    
    text=[]
    for t in input_text:
        if punc.search(t):
            continue
        text.append(t)
    stats = []
    stats.append(unique_words(text))
    stats.append(total_words(text))
    stats.append(lexical_diversity(text))
    stats.append(unusual_words(text))
    stats.append(content_fraction(text))
    stats.append(avg_len_words(text))
    names = ['unique_words', 'total_words','lex_div','unusual_words','content_frac','avg_len_words']
    text_joined = " ".join(text)
    ss = sid.polarity_scores(text_joined)
    for k in sorted(ss):
        stats.append(ss[k])
        names.append(k)
    return stats, names

def get_song_data(tokenized_song):
    
    lyrics = tokenized_song
    #get sentences without tokens
    lyrics_sep = []
    for y in lyrics:
        tmpsent = [word for (word,tag) in y]
        lyrics_sep = lyrics_sep + tmpsent

    if len(lyrics_sep)==0:
        return [], []

    x_data = []
    x_names = []
    tagged_lyrics = [j for i in lyrics for j in i]
    #get word
    stats, names = get_word_stats(lyrics_sep)
    x_names = x_names + names
    x_data = x_data + stats

    stats, names = get_pos_stats(tagged_lyrics)
    x_names = x_names + names
    x_data = x_data + stats
    return x_data, x_names


from flask import Flask, request, render_template, Markup

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html', reco_df='', display="none")


@app.route('/', methods=['POST'])
def main():
    
    ds = "/var/www/FlaskApp/FlaskApp/dataframe_storage.csv"

    user_song_name = request.form['song'] + " " + request.form['artist']
    test_lyric = search_musix_track(user_song_name)
    if test_lyric=="":
        return "oops, seems like the song's lyrics could not be found, insert artists name or try another song."

    tokenized_song = tokenize_song(test_lyric)
    user_data, x_names = get_song_data(tokenized_song)

    all_data = pd.read_csv(ds)

    if len(user_data)==0:
        return "oops, seems like the user tokenized song was not correct...error, contact me :)"

    user_data = np.array(user_data)
    user_data = user_data.reshape(1,-1)
    X_train, X_test, y_train, y_test, scaler= get_normalized_and_split_data(all_data, x_names,split=0.0)
    user_scaled_data= scaler.transform(user_data)
    
    b = get_euc_dist(user_scaled_data,X_train,[user_song_name],y_train,n_top=10)
    return render_template('index.html', song_name=user_song_name,
        reco_df=Markup(str(b).encode(encoding='UTF-8',errors='ignore')),  display="block")

if __name__ == '__main__':
    app.run(debug=True, port=80)
    main()
