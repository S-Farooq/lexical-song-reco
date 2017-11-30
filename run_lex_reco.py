from time import time
import numpy as np
import pprint

import matplotlib
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

pp = pprint.PrettyPrinter(indent=4)
class MyPrettyPrinter(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if isinstance(object, unicode):
            return (object.encode('utf8'), True, False)
        return pprint.PrettyPrinter.format(self, object, context, maxlevels, level)
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (20,10)

from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

import re
import requests  
from bs4 import BeautifulSoup
import nltk, pprint
import cPickle
from io import BytesIO  

np.random.seed(42)

def get_normalized_and_split_data(all_data,x_names,split=0.2,by_artist=False):
#     all_data = all_data[all_data['labels'].notnull()]
    if by_artist:
        artists = all_data['artist'].unique()
        print artists
        X_train = all_data[all_data['artist'].isin(artists[:-2])][x_names].values
        y_train = list(all_data[all_data['artist'].isin(artists[:-2])]['labels'])

        X_test = all_data[all_data['artist'].isin(artists[-2:])][x_names].values
        y_test = list(all_data[all_data['artist'].isin(artists[-2:])]['labels'])
    else:
        y = list(all_data['labels'])
        X = all_data[x_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=3)
    # print x_names
    #VARIANCE FILTER
    # model = ExtraTreesClassifier()
    # model.fit(X_train, y_train)
    # print(model.feature_importances_)

    # selector = VarianceThreshold(0.02)
    # X_train = selector.fit_transform(X_train)
    # X_test = selector.transform(X_test)

    # x_names = [all_data.columns[i] for i in selector.get_support(indices=True)]
    # print len(x_names), "out of", len(all_data.columns), "selected"
    # print x_names


    #STANDARD SCALING
    # scaler = MaxAbsScaler()
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
    # print to_drop
    # to_keep = np.array(set(range(X_train.shape[0])) - to_drop)


    X_train = np.delete(X_train, to_drop, axis=0)
    y_train = np.delete(y_train, to_drop, axis=0)
    # X_train = X_train[to_keep,:]
    # y_train = y_train[to_keep]

    n_samples, n_features = X_train.shape
    # labels = y_labels


    # print("n_samples %d, \t n_features %d"
    #       % (n_samples, n_features))
    return X_train, X_test, y_train, y_test, scaler


def get_fitted_clusters(X_train,X_test,y_train,y_test,x_names,user_song_name="",user_scaled_data=[],n_clusters = 5):
    n_clusters = 5

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
        
        # if len(y_test)>0:
        #     tmp_test = tmp[tmp['test']==1].sort_values(c,ascending=True)
        #     print "---TEST:", pp.pprint(list(tmp_test['labels'][:num]))
        
        # if len(user_scaled_data)>0:
        #     tmp_test = tmp[tmp['test']==2].sort_values(c,ascending=True)
        #     print "---USER_TEST:", pp.pprint(list(tmp_test['labels'][:num]))

    df_centroid = pd.DataFrame(kmeans.cluster_centers_)
    
    print"\n PLOT OF CLUSTER QUALITIES vs YOUR SONG:"
    df_centroid.columns = x_names
    if len(user_scaled_data)>0:
        df_tmp = pd.DataFrame([user_scaled_data])
        df_tmp.columns = x_names
        df_tmp.index = [user_song_name]
        df_centroid = df_centroid.append(df_tmp)
    df_centroid = df_centroid.transpose()

    ax1 = df_centroid.plot(x_compat=True, lw=3)
    fig = plt.xticks(range(len(df_centroid.index)), x_names, rotation=90)
    ax1.set_ylim(-4,4)
    plt.grid(True)
    plt.savefig('tmp_plot.png')

    return to_return.encode(encoding='UTF-8',errors='ignore')



def get_euc_dist(set1,set2,set1_y,set2_y,n_top=7):
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
    
    # threedee = plt.figure().gca(projection='3d')
    # threedee.scatter(ed_df_p['from_ind'], ed_df_p['to_ind'], ed_df_p['distance'])
    # threedee.set_xlabel('from')
    # threedee.set_ylabel('to')
    # threedee.set_zlabel('distance')
    # plt.show()

    cols = ['from','to','distance']
    ed_df = ed_df[cols]
    ed_df = ed_df.sort_values(['from','distance'],ascending=True)
    
    ed_df_top = ed_df.groupby('from').head(n_top)
    ed_df_top['rel_conf'] = ed_df_top['distance']/ed_df_top['distance'].max()
    ed_df_top['rel_conf'] = ed_df_top['rel_conf']-ed_df_top['rel_conf'].min()
    ed_df_top['rel_conf'] = (1.0-ed_df_top['rel_conf'])*100

    ed_df_top = ed_df_top.rename(columns={'to': 'My_Songs', 'rel_conf': 'Relative_Confidence'})
    msg = "<br>SONGS IN MY PLAYLIST CLOSEST (Euclidean distance) TO YOURS & RELATIVE CONFIDENCE:<br>"
    return msg+ ed_df_top[['My_Songs','Relative_Confidence']].sort_values(['Relative_Confidence'],ascending=False).to_html(index=False, header=False)

def search_musix_corpus(artists, pages=2,corpus_file='lyric_corpus.p'):
    p = re.compile('\/lyrics\/*')

    webpage = "https://www.musixmatch.com"
    url = 'https://www.musixmatch.com/search/'
    useragent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'

    try:
        lyric_corpus = cPickle.load(open(corpus_file, 'rb'))
    except:
        lyric_corpus = {}

    # artists = ["the national", "Editors", "chvrches", "iron and wine", "william fitzsimmons", "vienna teng"]
    # artists = ["oh wonder",'the shins','the killers','the strokes','bleachers']
#     artists = ["alvvays","andrew bird","birdy","bon iver","kings of leon"]
    # lyric_corpus = {}
    for a in artists:
        artist_lyrics = {}
        searchlink = url + a
        r  = requests.get(searchlink, headers={'User-Agent': useragent})
        soup = BeautifulSoup(r.content, "lxml")
        gdata = soup.find_all('a',{'class':'cover'})
        artist_page = webpage + str(gdata[0].get('href')) 


        for i in range(1,pages+1):
            artist_page_num = artist_page
            if i>1:
                artist_page_num = artist_page+"/"+str(i)
            # print artist_page_num

            try:
                r  = requests.get(artist_page_num, headers={'User-Agent': useragent})
            except:
                print "No lyrics page {i}, leaving this artist.".format(i=i)
                break
            soup = BeautifulSoup(r.content, "lxml")
            gdata = soup.find_all('a',{'class':'title'})

            for s in gdata:
                slink = str(s.get('href'))
                if p.search(slink):
                    song_lyrics = []
                    song_page = webpage+slink
    #                 print song_page
                    try:
                        r  = requests.get(song_page, headers={'User-Agent': useragent})
                    except:
                        print "cant access lyric page for {song_page},.".format(song_page=song_page)
                        continue
                    soup = BeautifulSoup(r.content, "lxml")
    #                 print soup
                    song_name = soup.find_all('h1',{'class':'mxm-track-title__track'})
                    song_name = song_name[0].text
                    song_name = song_name[6:] + "-" + a
                    print song_name.encode(encoding='UTF-8',errors='ignore')
                    lyrics = soup.find_all('p',{'class':'mxm-lyrics__content'})
    #                 print lyrics
                    if len(lyrics)==0:
                        print "No lyrics found for {song}".format(song=song_page)
                    for l in lyrics:
                        song_lyrics.append(l.text)

                    song_lyrics  = "\n".join(song_lyrics)
                    song_lyrics = song_lyrics.replace("\n\n","\n")
    #                 print song_lyrics
                    artist_lyrics[song_name]=song_lyrics
    #                 break

        lyric_corpus[a] = artist_lyrics

    cPickle.dump(lyric_corpus, open(corpus_file, 'wb'))
    return lyric_corpus

def search_musix_track(search_term):
    p = re.compile('\/lyrics\/*')

    webpage = "https://www.musixmatch.com"
    url = 'https://www.musixmatch.com/search/'
    useragent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'

    searchlink = url + search_term + "/tracks"
    try:
        r  = requests.get(searchlink, headers={'User-Agent': useragent})
    except:
        print "No track found...something went wrong.."
        return
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
            return
        soup = BeautifulSoup(r.content, "lxml")
#                 print soup
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
    from nltk.tag.perceptron import PerceptronTagger
    tagger = PerceptronTagger() 
    tagged = []
    sentences = document.split("\n")
#     return sentences
#     sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for sent in sentences:
        tagged_tokens = tagger.tag(sent)
        tagged.append(tagged_tokens)
    
    return tagged

def tokenize_song(lyrics):
    tokenized_lyrics = ie_preprocess(lyrics)
    return tokenized_lyrics

def tokenize_corpus(lyric_corpus_file, tokenized_corpus_file="tokenized_lyric_corpus.p"):
    try:
        tokenized_lyric_corpus = cPickle.load(open(tokenized_corpus_file, 'rb'))
    except:
        tokenized_lyric_corpus = {}

    lyric_corpus = cPickle.load(open(lyric_corpus_file, 'rb'))
    for a in lyric_corpus.keys():
        if not (a in tokenized_lyric_corpus):
            tokenized_lyric_corpus[a] = {}
        print "ARTIST:", a
        for s in lyric_corpus[a].keys():
            if s in tokenized_lyric_corpus[a]:
                continue
            lyrics = lyric_corpus[a][s]
            print "---SONG:",s
            tokenized_lyrics = ie_preprocess(lyrics)
            tokenized_lyric_corpus[a][s] = tokenized_lyrics
            cPickle.dump(tokenized_lyric_corpus, open(tokenized_corpus_file, 'wb'))
    
    return tokenized_lyric_corpus



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
#         print "invalid lyric data.."
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

def get_corpus_dataframe(tokenized_lyric_file, output_file="dataframe_storage.csv"):
    tokenized_lyric_corpus = cPickle.load(open(tokenized_lyric_file, 'rb'))
    X_data = []
    y_labels=[]
    y_artists=[]
    x_final_names = []
    for a in tokenized_lyric_corpus.keys():
        print "ARTIST:", a
        total_songs = len(tokenized_lyric_corpus[a].keys())
        curr_song=0
        for s in tokenized_lyric_corpus[a].keys():
            x_data, x_names = get_song_data(tokenized_lyric_corpus[a][s])
            if len(x_data)==0:
                continue
            x_final_names =x_names
            X_data.append(x_data)
            y_labels.append(s)
            y_artists.append(a)
#             curr_song+=1
  
    # print x_names
    # print X
    # print y_labels
    all_data = pd.DataFrame(X_data)
    all_data.columns=x_final_names
    all_data['labels']=y_labels
    all_data['artist']=y_artists
    all_data.to_csv(output_file,  encoding='utf-8')
    print all_data.head()
    return all_data


from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('my-form.html')


@app.route('/', methods=['POST'])
def main():
    
    tlc ="tokenized_lyric_corpus.p"
    lc='lyric_corpus.p'
    ds = "dataframe_storage.csv"
    artists = ["the national", "Editors", "chvrches", "william fitzsimmons", "vienna teng",
        "oh wonder",'the shins','the killers','the strokes','bleachers',
        "alvvays","andrew bird","birdy","bon iver","kings of leon"]

    # lyric_corpus = search_musix_corpus(artists,pages=7,corpus_file=lc)
    user_song_name = request.form['text']
    test_lyric = search_musix_track(user_song_name)

    tokenized_song = tokenize_song(test_lyric)
    all_data = pd.read_csv(ds)
    user_data, x_names = get_song_data(tokenized_song)

    if len(user_data)==0:
        print "oops, seems like the user tokenized song was not correct...error, contact me :)"


    X_train, X_test, y_train, y_test, scaler= get_normalized_and_split_data(all_data, x_names,split=0.0)
    user_scaled_data= scaler.transform(user_data)

    a = get_fitted_clusters(X_train,X_test,y_train,y_test,x_names,user_song_name,user_scaled_data,n_clusters = 5)

    b = get_euc_dist([user_scaled_data],X_train,[user_song_name],y_train,n_top=5)
    #+ """<a href="/img">Continue to cluster graph</a>"""
    return str(a).encode(encoding='UTF-8',errors='ignore') + str(b).encode(encoding='UTF-8',errors='ignore') 

if __name__ == '__main__':
    app.run(debug=True)
    main()
