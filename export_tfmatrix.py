from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

# Custom stop words, we need a custom list to include english stop words and some of the nonsense that is in all of the emails. This should improve the overall cluster quality. We can figure out which
# words are causing problems by looking at some of the overlapping common words in each cluster.

custom_stop_words = ['com', 'http', 'www', 'html', '08', '2008', 'watch']

# Combine predefined English stop words with custom ones
combined_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# dataset = pd.read_csv("archive/CEAS_08.csv")
dataset = pd.read_csv("archive/SpamAssasin.csv")
phishing_filtered = dataset[dataset["label"] == 1]
#test_df = phishing_filtered[:20000]  
test_df = phishing_filtered
test_df.loc[:, 'subject'] = test_df['subject'].fillna('')
test_df.loc[:, 'body'] = test_df['body'].fillna('')
tfidf_vectorizer = TfidfVectorizer(max_features=10000, 
                                   stop_words=combined_stop_words, 
                                   max_df=0.8, 
                                   min_df=5) 

X_body = tfidf_vectorizer.fit_transform(test_df['body']).todense()

# Add subject feature and combine matrices
tfidf_subject = TfidfVectorizer(stop_words=combined_stop_words)
X_subject = tfidf_subject.fit_transform(test_df['subject']).todense()


tfidf_matrix = np.hstack([X_body, X_subject])


all_words = list(tfidf_vectorizer.get_feature_names_out()) + list(tfidf_subject.get_feature_names_out())
matrix_DataFrame = pd.DataFrame(tfidf_matrix, columns=all_words)

try:
     matrix_DataFrame.to_csv('tf_idfmatrix.csv', index=False)

except:
     print("Error exporting dataframe to CSV file")

