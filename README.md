"# lexical-song-reco" 

I tried a couple of models and techniques but settled on something quite basic: Euclidean Distance calculations. The idea is to generate feature vectors of lexical data from the lyrics of the user’s song and songs from the select playlist. Then, I calculate the euclidean distance between the 2 sets (the users vs. selected playlist) and present the closest distances to the user.

Analytical Features (Python, NLTK, Scikit-learn, Web-scraping)
Creating a Corpus: This would be the set of lyrics from my own playlist that were automatically found by web-scraping them off of a popular lyric website. (This is also how I get the user’s song lyrics).
Feature Extraction: To keep is simple, there were 3 sets of features I focused on — lexical/word stats, sentiment, and Parts-of-Speech ratios. Lexical stats include data like # of unique words, lexical diversity, # of unusual words, average length of words, etc. What these stats provide us is an overview of— in crude terms — the ‘intellect’ level of the song. Sentiment analysis extracted the amount of positive, neutral, and negative words or bi-grams in the lyrics. Lastly, Parts-of-Speech ratios (ex. how many nouns, verbs, adverbs, pronouns, etc.) helped to define perspectives and qualities of the song — this could include if the song was personal, if it involved a lot of actions, if it contained a lot of references, its general ‘vibe’, etc. All features were scaled using Standard Scaling (how many standard deviations they differed from the mean, which was calculated off my corpus). All this was done through simple math and the NLTK python library.
Distance Calculations: As i mentioned before, a simple euclidean distance measure between vectors.
Web-related Features (Flask, Spotify API, Chart JS)
Create a Spotify Playlist: It took me some time playing around with the Spotify API so that users could create automatically create a Spotify playlist from my recommendations — assuming they have playlist account privileges.
Graphing: I used some JavaScript libraries to display semi-interactive charts that help users visualize the data.
