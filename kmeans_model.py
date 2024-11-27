import pandas as pd
import nltk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from export_tfmatrix import test_df

#nltk.download('punkt')

# Tokenize?
# k = 6 with regular english stopwords was pretty good. 
# k = 10 with the added stop words is pretty good, but there is still a lot of overlap between clusters 4, 5, and 6
# The original, 20,000 sample filtered dataset (CEAS08.csv) did pretty well with this code. It just had a weird inertia plot. 
n_clusters = 9
tf_idf_df = pd.read_csv("tf_idfmatrix.csv")  


# Rename weird columns
# for col in tf_idf_df.columns:
#     if col.endswith(".1"):

 
#         newname = col[:len(col)-2]
#         tf_idf_df[newname] = tf_idf_df[col].copy()
#         tf_idf_df.drop(col, axis=1, inplace=True)
#         print(f"Renaming column {col} to {newname}")


        #  temp = tf_idf_df[col]
        # print(temp)
        # tf_idf_df.drop(col, axis=1, inplace=True)
        # newname = col[:len(col)-2]
        # tf_idf_df[newname] = temp
        # print(f"Renaming column {col} to {newname}")
        
#changed max_iter to 500?
kmeans = KMeans(n_clusters=n_clusters, random_state=90215, max_iter=500)
kmeans.fit(tf_idf_df)

labels = kmeans.labels_


# Code for WordCLoud visualization
# spam_words = ' '.join(list(tf_idf_df.columns))
# spam_wordcloud = WordCloud(width=800, height=400, background_color='black', max_words=200).generate(spam_words)

# plt.figure(figsize=(10, 8))
# plt.imshow(spam_wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()



# Inertia/elbow method - We chose our k based on this and the most common words in each cluster (whichever made the most sense)
# distorsions = []
# for k in range(2,20):
#     kmeans = KMeans(n_clusters=k, random_state=90215, max_iter=500)
#     kmeans.fit(tf_idf_df)
#     distorsions.append(kmeans.inertia_)

# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(2, 20), distorsions)
# plt.xticks(range(2, 20))
# plt.grid(True)
# plt.title('Elbow Curve')
# plt.xlabel("K")
# plt.ylabel("Inertia")
# plt.show()

# Print the 15 most common words in each cluster
# dataframe_copy = test_df.copy()
# dataframe_copy.loc[:, 'Cluster'] = labels
# for cluster_num in range(n_clusters):

#     cluster_docs = tf_idf_df[labels == cluster_num]

#     mean_tfidf = cluster_docs.mean(axis=0).sort_values(ascending=False)
    
#     top_words = mean_tfidf.head(15)
#     plt.figure(figsize=(10, 6))
#     plt.barh(top_words.index, top_words.values, color='skyblue')
#     plt.gca().invert_yaxis()  
#     plt.title(f"Top 15 Words in Cluster {cluster_num + 1}")
#     plt.xlabel("Mean TF-IDF Score")
#     with open(f'cluster_samples/cluster{cluster_num + 1}_examples.txt','w', encoding='utf-8') as outfile:
#         dataframe_copy[dataframe_copy['Cluster'] == cluster_num][:20].to_string(outfile,columns=['subject']) #could add body, but many are inappropriate and too long.
#     outfile.close()
#     plt.savefig(f'images/topwords_cluster{cluster_num + 1}.png')
#     plt.show()


# Visualize clusters using scatterplot, we need to reduce dimensions (thousands) using tsne before plotting otherwise it would be impossible to visualize
# We could try increasing max_iter?
# tsne = TSNE(n_components=2, random_state=90215, perplexity=30, max_iter=300)
# reduced_data = tsne.fit_transform(tf_idf_df)


# reduced_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
# reduced_df['Cluster'] = labels

# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x='Component 1', y='Component 2',
#     hue='Cluster', data=reduced_df,
#     palette='viridis', legend='full', s=60
# )
# plt.title("KMeans Clusters (t-SNE reduced TF-IDF data)")
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.legend(title='Cluster')
# plt.show()