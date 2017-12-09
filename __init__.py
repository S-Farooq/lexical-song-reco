

from flask import Flask, request, redirect, g, render_template, Markup, session, url_for
from util_fxns import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import base64
import urllib, difflib

# Authentication Steps, paramaters, and responses are defined at https://developer.spotify.com/web-api/authorization-guide/
# Visit this url to see all the steps, parameters, and expected response. 


app = Flask(__name__)
app.secret_key = '5f535ebef7444444gb42d58590161e7bfcf653'
#  Client Keys
CLIENT_ID = "5f535ebef74b42d58590161e7bfcf653"
CLIENT_SECRET = "b9f8b9f7c055433894bdcef35a2905f0"

# Spotify URLS
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)


# Server-side Parameters
CLIENT_SIDE_URL = "http://songreco.shaham.me"
PORT = 8080
REDIRECT_URI = "{}/callback/q".format(CLIENT_SIDE_URL)
SCOPE = "playlist-modify-public playlist-modify-private"
STATE = ""
SHOW_DIALOG_bool = True
SHOW_DIALOG_str = str(SHOW_DIALOG_bool).lower()


auth_query_parameters = {
    "response_type": "code",
    "redirect_uri": REDIRECT_URI,
    "scope": SCOPE,
    # "state": STATE,
    # "show_dialog": SHOW_DIALOG_str,
    "client_id": CLIENT_ID
}

corpus_dict = {
        'Top songs by my artists': "dataframe_storagewpop2",
    }

def get_mrkup_from_df(reco_df,to_display_amount=10):
    reco_mrkup = ["""<table class="table table-hover"><thead><tr>
        <th>{columns}</th></tr></thead><tbody>
      """.format(columns="</th><th>".join(reco_df.columns))]

    for index, row in reco_df.iterrows():
        if to_display_amount==0:
            break
        to_display_amount = to_display_amount - 1
        row = [str(x).upper() for x in row]
        reco_mrkup.append("""<tr>
        <th>{vals}</th></tr>
            """.format(vals="</th><th>".join(row)))

    reco_mrkup.append("""</tbody></table>""")
    reco_display = "\n".join(reco_mrkup)
    return reco_display

def auth_spot():
    # Auth Step 1: Authorization
    url_args = "&".join(["{}={}".format(key,urllib.quote(val)) for key,val in auth_query_parameters.iteritems()])
    auth_url = "{}/?{}".format(SPOTIFY_AUTH_URL, url_args)
    return auth_url


@app.route("/callback/q")
def callback():
    # Auth Step 4: Requests refresh and access tokens
    auth_token = request.args['code']
    code_payload = {
        "grant_type": "authorization_code",
        "code": str(auth_token),
        "redirect_uri": REDIRECT_URI
    }
    base64encoded = base64.b64encode("{}:{}".format(CLIENT_ID, CLIENT_SECRET))
    headers = {"Authorization": "Basic {}".format(base64encoded)}
    post_request = requests.post(SPOTIFY_TOKEN_URL, data=code_payload, headers=headers)

    # Auth Step 5: Tokens are Returned to Application
    response_data = json.loads(post_request.text)
    access_token = response_data["access_token"]
    refresh_token = response_data["refresh_token"]
    token_type = response_data["token_type"]
    expires_in = response_data["expires_in"]

    # Auth Step 6: Use the access token to access Spotify API
    authorization_header = {"Authorization":"Bearer {}".format(access_token)}

    post_header = {"Authorization":"Bearer {}".format(access_token), "Content-Type": "application/json"}

    # Get profile data
    user_profile_api_endpoint = "{}/me".format(SPOTIFY_API_URL)
    profile_response = requests.get(user_profile_api_endpoint, headers=authorization_header)
    profile_data = json.loads(profile_response.text)

    # Create Playlist
    try:
        usong =session['usong'].upper()
        uartist =session['uartist'].upper()
        playlist_info = {
            "name": "Lex-Recos based on "+usong,
            "description": "A playlist consisting of Shaham's songs that are lexically similar to {song} by {artist}.".format(song=usong,artist=uartist)
        }
        playlist_api_endpoint = "{}/playlists".format(profile_data["href"])
        post_request = requests.post(playlist_api_endpoint, data=json.dumps(playlist_info), headers=post_header)
        print post_request.text
        response_data = json.loads(post_request.text)
        
        #playlist vars
        playlist_id = response_data['id']
        playlist_url = response_data['external_urls']['spotify']
    except:
        session['callback_playlist'] = response_data
        return redirect(url_for('.my_form'))
    
    to_display_amount=25
    reco_df =pd.read_json(session['reco_df'], orient='split')
    to_display = []
    uri_list=[]
    for index, row in reco_df.iterrows():
        if to_display_amount==0:
            break
        to_display_amount = to_display_amount - 1
        #search track
        try:
            song_to_search=re.sub(r'[^a-zA-Z0-9\s]', '', str(row['My Song']).lower())
            song_to_search=re.sub(r'\s+', '+', song_to_search)
            track_search_api_endpoint = "{}/search?q={}&type=track".format(SPOTIFY_API_URL,song_to_search)
            search_response = requests.get(track_search_api_endpoint, headers=authorization_header)
            search_data = json.loads(search_response.text)
            if len(search_data['tracks']['items'])==0:
                continue
            
            artist_choices=[]
            for t in search_data['tracks']['items']:
                artist_choices.append(t['artists'][0]['name'].upper())
            
            closest_artists = difflib.get_close_matches(str(row['Artist']).upper(), artist_choices,1)

            # to_display.append("<p>"+str(row['Artist']).upper() + "-"+str(artist_choices)+"<br></p>")
            if len(closest_artists)>0:
                closest_artist = closest_artists[0]
                for t in search_data['tracks']['items']:
                    if t['artists'][0]['name'].upper()==closest_artist:
                        uri_list.append(t['uri'])
                        break
            # else:
            #     uri_list.append(search_data['tracks']['items'][0]['uri'])
        except:
            continue

    #ADD list of uris to playlist (add tracks)
    try:
        add_track_api_endpoint = "{}/playlists/{}/tracks".format(profile_data["href"],playlist_id)
        track_data = {
            "uris": uri_list,
        }
        post_request = requests.post(add_track_api_endpoint, data=json.dumps(track_data), headers=post_header)
        response_data = json.loads(post_request.text)
    except:
        session['callback_playlist'] = response_data + str(track_data['uris'])
        return redirect(url_for('.my_form'))       
    

    # Get user playlist data
    # playlist_api_endpoint = "{}/playlists".format(profile_data["href"])
    # playlists_response = requests.get(playlist_api_endpoint, headers=authorization_header)
    # playlist_data = json.loads(playlists_response.text)
    
    # Combine profile and playlist data to display
    # display_arr = [profile_data] + playlist_data["items"]
    session['callback_playlist'] = Markup("<a href='{playlist_url}' target='_blank'><h3>Your New Lex-Recos Playlist</h3></a>".format(playlist_url=playlist_url))
    # session['callback_playlist'] = Markup("".join(to_display))
    # reco_df =pd.read_json(session['reco_df'], orient='split')
    # usong =session['usong']
    # uartist =session['uartist']
    # reco_display = get_mrkup_from_df(reco_df,to_display_amount=2)

    return redirect(url_for('.my_form'))
    # return redirect(url_for('.main', 
    #         song_name=usong.upper(), artist_name=uartist.upper(),
    #         reco_df=Markup(str(reco_display).encode(encoding='UTF-8',errors='ignore') + pprint.pformat(display_arr, indent=4)),  display="block"))

    # return redirect('index.html',
    #         song_name=usong.upper(), artist_name=uartist.upper(),
    #         reco_df=Markup(str(reco_display).encode(encoding='UTF-8',errors='ignore')),  display="block")
    # return render_template("index.html", reco_df=display_arr)

@app.route('/')
def my_form():
    if 'callback_playlist' in session:
        reco_df =pd.read_json(session['reco_df'], orient='split')
        usong =session['usong']
        uartist =session['uartist']
        full_reco_df=session['user_song_values']
        x_names=session['features']
        colors=session['colors']
        reco_display = get_mrkup_from_df(reco_df,to_display_amount=7)
        to_show_reco=Markup(str(reco_display).encode(encoding='UTF-8',errors='ignore')) 
        session.clear()
        if 'callback_playlist' in session:
            return render_template('index.html', scroll="recos",
                song_name=usong.upper(), artist_name=uartist.upper(),
                reco_df=to_show_reco,  display="block",corpus_dict=corpus_dict,
                user_song_values=full_reco_df,features=x_names,colors=colors
                )
        else:
            return render_template('index.html', scroll="recos",
                song_name=usong.upper(), artist_name=uartist.upper(),
                reco_df=to_show_reco,  display="block", corpus_dict=corpus_dict,
                user_song_values=full_reco_df,features=x_names,colors=colors)
    else:
        return render_template('index.html', corpus_dict=corpus_dict)


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.form['btn'] == 'search':
        try:
            session.clear()
            
            dbase = request.form['dbase']
            csv_file = corpus_dict[dbase]
            ds = "/var/www/FlaskApp/FlaskApp/{csv_file}.csv".format(csv_file=csv_file)
            
            usong=request.form['song']
            uartist=request.form['artist']

            session['usong']=usong
            session['uartist']=uartist
            
            if usong=="" or uartist=="":
                return render_template('index.html', display_alert="block", corpus_dict=corpus_dict,
                    err_msg="Please enter a song & artist to match against...")

            user_song_name = usong + " " + uartist
            
            try:
                test_lyric = search_musix_track(user_song_name)
            except Exception as e:
                err_msg = str(e) + ".oops, seems like the song's lyrics could not be found, please try another song...or contact me :)"
                return render_template('index.html', corpus_dict=corpus_dict,display_alert="block", err_msg=err_msg)

            tokenized_song = tokenize_song(test_lyric)
            user_data, x_names = get_song_data(tokenized_song)

            all_data = pd.read_csv(ds, encoding="utf-8")

            if len(user_data)==0 or user_data[0]==0.0:
                return render_template('index.html', display_alert="block", corpus_dict=corpus_dict,
                    err_msg="oops, seems like the song could not be analyzed correctly...error, contact me :)")

            user_data = np.array(user_data)
            user_data = user_data.reshape(1,-1)
            X_train, X_test, y_train, y_test, scaler= get_normalized_and_split_data(all_data, x_names,split=0.0)
            user_scaled_data= scaler.transform(user_data)
            
            reco_df, full_reco_df = get_euc_dist(user_scaled_data,X_train,[user_song_name],y_train,x_names,n_top=25)
            session['reco_df']=reco_df.to_json(orient='split')
            
            

            reco_display = get_mrkup_from_df(reco_df,to_display_amount=7)
            num_to_graph=7
            full_reco_df = full_reco_df.head(num_to_graph)
            full_reco_df = full_reco_df[["My Songs"] +x_names].values.tolist()
            full_reco_df.append([usong.upper()+"-"+uartist.upper()]+user_scaled_data[0,:].tolist())
            import random
            r = lambda: random.randint(0,255)
            colors=[]
            for i in range(num_to_graph):
                colors.append('%02X%02X%02X' % (r(),r(),r()))

            session['user_song_values']=full_reco_df
            session['features']=x_names
            session['colors']=colors
            return render_template('index.html', scroll="recos", 
                song_name=usong.upper(), artist_name=uartist.upper(),
                reco_df=Markup(str(reco_display).encode(encoding='UTF-8',errors='ignore')),  display="block",corpus_dict=corpus_dict,
                user_song_values=full_reco_df,features=x_names,colors=colors)
        except Exception as e:
            err_msg = str(e) + "ERROR: Sorry, looks like something has gone wrong... shoot me a message and I'll try to fix it!"
            return render_template('index.html', display_alert="block", err_msg=err_msg,corpus_dict=corpus_dict)
    elif request.form['btn'] == 'search_custom':
        try:
            session.clear()
            dbase = request.form['dbase']
            csv_file = corpus_dict[dbase]
            ds = "/var/www/FlaskApp/FlaskApp/{csv_file}.csv".format(csv_file=csv_file)

            usong="Custom Text"
            uartist="Your Input"

            session['usong']=usong
            session['uartist']=uartist


            user_song_name = usong + " " + uartist
            
            if len(request.form['custom_text'])<100:
                return render_template('index.html', display_alert="block", corpus_dict=corpus_dict,
                    err_msg="oops, please enter at least 100 or more characters for a valid analysis!")

            test_lyric = get_custom_text_lyric(request.form['custom_text'])

            tokenized_song = tokenize_song(test_lyric)
            user_data, x_names = get_song_data(tokenized_song)

            all_data = pd.read_csv(ds, encoding="utf-8")

            if len(user_data)==0 or user_data[0]==0.0:
                return render_template('index.html', display_alert="block", corpus_dict=corpus_dict,
                    err_msg="oops, seems like the song could not be analyzed correctly...error, contact me :)")
                
            user_data = np.array(user_data)
            user_data = user_data.reshape(1,-1)
            X_train, X_test, y_train, y_test, scaler= get_normalized_and_split_data(all_data, x_names,split=0.0)
            user_scaled_data= scaler.transform(user_data)
            
            reco_df, full_reco_df = get_euc_dist(user_scaled_data,X_train,[user_song_name],y_train,x_names,n_top=25)

            session['reco_df']=reco_df.to_json(orient='split')
            
            
            reco_display = get_mrkup_from_df(reco_df,to_display_amount=10)
            num_to_graph=7
            full_reco_df = full_reco_df.head(num_to_graph)
            full_reco_df = full_reco_df[["My Songs"] +x_names].values.tolist()
            full_reco_df.append([usong.upper()+"-"+uartist.upper()]+user_scaled_data[0,:].tolist())
            import random
            r = lambda: random.randint(0,255)
            colors=[]
            for i in range(num_to_graph):
                colors.append('%02X%02X%02X' % (r(),r(),r()))

            session['user_song_values']=full_reco_df
            session['features']=x_names
            session['colors']=colors

            # user_scaled_data = user_scaled_data[0,:].tolist()
            return render_template('index.html', scroll="recos", 
                song_name=usong.upper(), artist_name=uartist.upper(),
                reco_df=Markup(str(reco_display).encode(encoding='UTF-8',errors='ignore')),  display="block",corpus_dict=corpus_dict,
                user_song_values=full_reco_df,features=x_names,colors=colors)
        except Exception as e:
            err_msg = str(e) + "ERROR: Sorry, looks like something has gone wrong... shoot me a message and I'll try to fix it!"
            return render_template('index.html', display_alert="block", err_msg=err_msg,corpus_dict=corpus_dict)

    elif request.form['btn'] == 'playlist':
        return redirect(auth_spot())
    elif request.form['btn'] == 'more':
        reco_df =pd.read_json(session['reco_df'], orient='split')
        usong =session['usong']
        uartist =session['uartist']
        reco_display = get_mrkup_from_df(reco_df,to_display_amount=25)

        return render_template('index.html',
            song_name=usong.upper(), artist_name=uartist.upper(),
            reco_df=Markup(str(reco_display).encode(encoding='UTF-8',errors='ignore')),  display="block",corpus_dict=corpus_dict)
    else:
        return render_template("index.html",corpus_dict=corpus_dict)

if __name__ == '__main__':
    app.run(debug=True, port=80)
    main()
