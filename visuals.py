#find email sample from cluster that doesn't make sense and display it
#heatmap
#more scatterplots


# HEAT PLOTSSSSSSSSSSSSSSSSSSSS

# DOESN'T WORK
#Heatmap for clusters

# sns.heatmap(
#     np.array(metrics).reshape(-1, 1),  # Reshape to 2D for heatmap
#     annot=True,
#     xticklabels=['Silhouette Score'],
#     yticklabels=k_values,
#     cmap='coolwarm'
# )

# # Customize plot
# plt.title('Clustering Metrics Heatmap')
# plt.ylabel('Number of Clusters (k)')
# plt.xlabel('Metric')
# plt.show()


# WORKS
# cluster_counts = pd.Series(labels).value_counts().sort_index()

# # Prepare data for heatmap
# heatmap_data = cluster_counts.values.reshape(-1, 1)  # Reshape for heatmap

# # Plotting the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     heatmap_data,
#     annot=True,
#     fmt='d',
#     cmap='coolwarm',
#     xticklabels=['Cluster Size'],
#     yticklabels=[f'Cluster {i}' for i in range(n_clusters)]
# )

# # Customizing the plot
# plt.title('Cluster Sizes Heatmap')
# plt.ylabel('Clusters')
# plt.xlabel('Metric')
# plt.show()   



# DOESN'T WORK
# heatmap_data = pd.DataFrame(0, index=tf_idf_df.index, columns=[f'Cluster {i}' for i in range(n_clusters)])
# for i, label in enumerate(labels):
#     heatmap_data.iloc[i, label] = 1

# # Debugging: Print heatmap_data to ensure binary values
# print(heatmap_data.head())

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     heatmap_data,
#     cmap='coolwarm',  # Switch to 'YlGnBu' or 'binary' for a different effect
#     cbar_kws={'label': 'Cluster Membership'},
#     linewidths=0.5,
#     linecolor='lightgray',
#     vmin=0, vmax=1  # Ensures binary scale (0 or 1)
# )

# # Customizing the plot
# plt.title('Cluster Membership Heatmap', fontsize=16, fontweight='bold')
# plt.xlabel('Clusters', fontsize=14)
# plt.ylabel('Samples', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks([], fontsize=12)  # Hide y-tick labels for clarity
# plt.show()



# SCATTER PLOTSSSSSSSSSSSSSSSSSSS


# WORKS
# pca = PCA(n_components=2, random_state=42)
# reduced_data = pca.fit_transform(tf_idf_df)

# # Create a DataFrame for visualization
# scatter_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
# scatter_df['Cluster'] = labels

# # Plot the clusters
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x='PC1', 
#     y='PC2', 
#     hue='Cluster', 
#     palette='tab10',  # Use 'tab20', 'viridis', or others for variety
#     data=scatter_df,
#     s=100,  # Marker size
#     alpha=0.8  # Transparency for overlapping points
# )

# # Customize plot aesthetics
# plt.title('K-Means Clustering (PCA Projection)', fontsize=16, fontweight='bold')
# plt.xlabel('Principal Component 1', fontsize=14)
# plt.ylabel('Principal Component 2', fontsize=14)
# plt.legend(title='Cluster', fontsize=12, loc='upper right')
# plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
# plt.show()





# WORKS
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# tsne_data = tsne.fit_transform(tf_idf_df)

# # Create a DataFrame for visualization
# scatter_df_tsne = pd.DataFrame(tsne_data, columns=['TSNE1', 'TSNE2'])
# scatter_df_tsne['Cluster'] = labels

# # Plot the clusters
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x='TSNE1', 
#     y='TSNE2', 
#     hue='Cluster', 
#     palette='Spectral',  # Colorful palette
#     data=scatter_df_tsne,
#     s=100, 
#     alpha=0.8
# )

# # Customize plot aesthetics
# plt.title('K-Means Clustering (t-SNE Projection)', fontsize=16, fontweight='bold')
# plt.xlabel('t-SNE Dimension 1', fontsize=14)
# plt.ylabel('t-SNE Dimension 2', fontsize=14)
# plt.legend(title='Cluster', fontsize=12, loc='upper right')
# plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
# plt.show()



# WORKS
# Combine reduced PCA data and cluster labels
# pca = PCA(n_components=2, random_state=42)
# reduced_data = pca.fit_transform(tf_idf_df)

# # Create a DataFrame for visualization
# scatter_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
# scatter_df['Cluster'] = labels

# # Create a pairplot
# sns.pairplot(
#     scatter_df, 
#     hue='Cluster', 
#     palette='tab10',
#     diag_kind='kde',  # Kernel density estimation on the diagonal
#     corner=True  # Plot only the lower triangle of the grid
# )

# # Customize plot
# plt.suptitle('Pairwise Relationships of Clusters', y=1.02, fontsize=16, fontweight='bold')
# plt.show()


# BAR PLOTSSSSSSSSSSSSSS


# WORKS
# Count the number of points in each cluster
# cluster_sizes = pd.Series(labels).value_counts().sort_index()

# # Plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette="tab10")

# # Customize
# plt.title('Cluster Sizes', fontsize=16, fontweight='bold')
# plt.xlabel('Cluster', fontsize=14)
# plt.ylabel('Number of Points', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.5)
# plt.show()


# KINDA WORKS - ASK PATRICK

# Add cluster labels to the data
# tf_idf_df['Cluster'] = labels

# # Calculate average values per cluster
# cluster_means = tf_idf_df.groupby('Cluster').mean()

# # Plot
# plt.figure(figsize=(12, 8))
# sns.heatmap(cluster_means.T, cmap='coolwarm', annot=False, cbar=True, linewidths=0.5)

# # Customize
# plt.title('Average Feature Values per Cluster', fontsize=16, fontweight='bold')
# plt.xlabel('Cluster', fontsize=14)
# plt.ylabel('Features', fontsize=14)
# plt.show()



# KINDA WORKS - ASK PATRICK

# Example: Aggregate some meaningful feature groups
# feature_groups = tf_idf_df.iloc[:, :5].copy()  # Use the first 5 features as an example
# feature_groups['Cluster'] = labels
# cluster_sums = feature_groups.groupby('Cluster').sum()

# # Plot a stacked bar chart
# cluster_sums.T.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')

# # Customize
# plt.title('Feature Contributions by Cluster', fontsize=16, fontweight='bold')
# plt.xlabel('Features', fontsize=14)
# plt.ylabel('Sum of Feature Values', fontsize=14)
# plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
# plt.tight_layout()
# plt.show()


# WORKS 
# Normalize cluster sizes to get proportions
# cluster_sizes = pd.Series(labels).value_counts().sort_index()
# cluster_proportions = cluster_sizes / cluster_sizes.sum()

# # Plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x=cluster_proportions.index, y=cluster_proportions.values, palette="coolwarm")

# # Customize
# plt.title('Proportion of Points per Cluster', fontsize=16, fontweight='bold')
# plt.xlabel('Cluster', fontsize=14)
# plt.ylabel('Proportion', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.5)
# plt.show()

