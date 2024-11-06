from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

dataset = pd.read_csv("archive/CEAS_08.csv")
phishing_filtered = dataset[dataset["label"] == 1]
test_df = phishing_filtered[:1000]  

tfidf_vectorizer = TfidfVectorizer(max_features=10000, 
                                   stop_words='english', 
                                   max_df=0.8, 
                                   min_df=5) 


tfidf_matrix = tfidf_vectorizer.fit_transform(test_df['body'])

dense_matrix = tfidf_matrix.todense()
words = tfidf_vectorizer.get_feature_names_out()
matrix_DataFrame = pd.DataFrame(dense_matrix, columns=words)

try:
     matrix_DataFrame.to_csv('tf_idfmatrix.csv', index=False)

except:
     print("Error exporting dataframe to CSV file")

