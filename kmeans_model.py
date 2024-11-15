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

#nltk.download('punkt')
# Need to filter the body of some samples the original dataset (whitespace, etc), need to remove random words that are meaningless (www, hex strings, etc)
# Increase dataset subset size and see how that affects clusters
# Try using different features
# Tokenize?
# k = 6 with regular english stopwords was pretty good. 
# k = 10 with the added stop words is pretty good, but there is still a lot of overlap between clusters 4, 5, and 6
# Adding "watch" to the stop words list helps a lot
n_clusters = 10
tf_idf_df = pd.read_csv("tf_idfmatrix.csv")  

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tf_idf_df)

labels = kmeans.labels_


# Inertia/elbow method - We chose our k based on this and the most common words in each cluster (whichever made the most sense)
# distorsions = []
# for k in range(2, 20):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(tf_idf_df)
#     distorsions.append(kmeans.inertia_)

# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(2, 20), distorsions)
# plt.xticks(range(2, 20))
# plt.grid(True)
# plt.title('Elbow curve')
# plt.xlabel("K")
# plt.ylabel("Inertia")
# plt.show()

# Print the 10 most common words in each cluster
for cluster_num in range(n_clusters):

    cluster_docs = tf_idf_df[labels == cluster_num]

    mean_tfidf = cluster_docs.mean(axis=0).sort_values(ascending=False)
    
    top_words = mean_tfidf.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_words.index, top_words.values, color='skyblue')
    plt.gca().invert_yaxis()  
    plt.title(f"Top Words in Cluster {cluster_num}")
    plt.xlabel("Mean TF-IDF Score")
    plt.show()


# Visualize clusters using scatterplot, we need to reduce dimensions (thousands) using tsne before plotting otherwise it would be impossible to visualize
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
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