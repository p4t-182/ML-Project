import pandas as pd
import nltk
nltk.download('punkt_tab') 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv("archive/CEAS_08.csv")
phishing_filtered = dataset[dataset["label"] == 1]
test_df = phishing_filtered[:1000]  # Using a smaller subset for faster processing

# Step 2: Preprocess the text data (tokenize)
# Tokenize the email bodies
email_corpus = test_df['body'].apply(word_tokenize)

# Step 3: Train the Word2Vec model
model = Word2Vec(sentences=email_corpus, vector_size=100, window=5, min_count=5, workers=4)

# Step 4: Define a function to convert emails to vectors by averaging the word vectors
def get_email_vector(email):
    words = word_tokenize(email)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

# Step 5: Apply this function to all emails in the subset
email_vectors = test_df['body'].apply(get_email_vector)

# Convert email_vectors into a matrix for dimensionality reduction
email_vectors_matrix = np.vstack(email_vectors)

# Step 6: Reduce dimensions using PCA or t-SNE for visualization
# Using PCA for initial dimensionality reduction
pca = PCA(n_components=50)
reduced_vectors_pca = pca.fit_transform(email_vectors_matrix)

# Now use t-SNE for further dimensionality reduction to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors_tsne = tsne.fit_transform(reduced_vectors_pca)

# Step 7: Visualize the reduced email vectors (2D)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_vectors_tsne[:, 0], y=reduced_vectors_tsne[:, 1], hue=test_df['label'], palette="viridis", s=50)
plt.title('t-SNE Visualization of Email Similarity', fontsize=16)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.legend(title='Label', loc='best')
plt.show()

# dataset = pd.read_csv("archive/CEAS_08.csv")

# phishing_filtered = dataset[dataset["label"] == 1]

# test_df = phishing_filtered[:1000]


# # Step 1: Vectorize the email bodies using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')  # Adjust max_features for efficiency
# tfidf_matrix = tfidf_vectorizer.fit_transform(test_df['body'])
# print(type(tfidf_matrix))
# dense = tfidf_matrix.todense()
# # print(dense.shape)
# words = tfidf_vectorizer.get_feature_names_out()

# # # Create a Pandas DataFrame from the dense matrix
# df_tfidf = pd.DataFrame(dense, columns=words)
# max_tfidf_per_word = df_tfidf.max(axis=0)

# # Sort the scores in descending order
# sorted_tfidf_scores = max_tfidf_per_word.sort_values(ascending=False)

# # Show the top 10 words with the highest TF-IDF scores
# top_n = 10
# top_words = sorted_tfidf_scores.head(top_n)
# print(top_words)

# top_words.plot(kind='barh', figsize=(10, 6), color='skyblue')
# plt.xlabel('TF-IDF Score')
# plt.title(f'Top {top_n} Words with Highest TF-IDF Scores')
# plt.gca().invert_yaxis()  # Invert y-axis to show the largest score on top
# plt.show()
# # # print(len(test_df.iloc[0]['body']))
# # # print(test_df.iloc[1]['body'])
# # # Sample the first 20 emails and 50 most important words for visualization
# # subset_matrix = df_tfidf.iloc[:20, :500]

# # # Create a heatmap
# # plt.figure(figsize=(15, 8))
# # sns.heatmap(subset_matrix, cmap="YlGnBu", linewidths=.5)
# # plt.title('TF-IDF Heatmap (First 20 Emails and 50 Words)')
# # plt.show()