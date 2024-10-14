**Analysis Report**

**1\. Retrieving and Loading the Olivetti Faces Dataset**

The Olivetti Faces dataset, a standard benchmark for facial recognition tasks, consists of 400 grayscale images of 40 individuals, with 10 images per person. The images are preprocessed and reduced to 64×6464 \\times 6464×64 pixels, making them suitable for machine learning tasks like classification and clustering. The dataset was loaded using the fetch_olivetti_faces() function from sklearn.

**2\. Data Splitting**

The dataset was split into three sets:

- **Training set**: 70% of the data
- **Validation set**: 15% of the data
- **Test set**: 15% of the data

Stratified sampling was used to ensure that each set contains an equal number of images per person. This was important for preserving the distribution of the dataset across individuals and preventing bias toward specific subjects during the training or evaluation phases.

**3\. Training a Classifier Using K-fold Cross Validation**

An **SVM classifier** with a linear kernel was trained using **Stratified K-Fold cross-validation** with 5 splits. Cross-validation was used to evaluate the classifier's ability to predict the person represented in each picture. The following steps were taken:

- The training data was scaled using StandardScaler to ensure that features have zero mean and unit variance.
- The classifier was trained on each fold, and its performance was evaluated on the corresponding validation fold using **accuracy** as the metric.

The average accuracy across the 5 folds was computed to assess the classifier’s performance, which yielded competitive results for the task of identifying individuals based on facial images.

**4\. Dimensionality Reduction Using Hierarchical Clustering**

In this step, we applied **Agglomerative Hierarchical Clustering (AHC)** using three different similarity measures to reduce the dimensionality of the dataset. The clustering was performed using the **centroid-based clustering rule** (linkage type: average), and the silhouette score was used to assess cluster quality.

**4(a) Clustering Using Euclidean Distance**

The Euclidean distance metric clusters data points based on physical distance. In this method:

- Agglomerative Clustering was applied, merging clusters based on the minimum Euclidean distance between points.
- The silhouette score was used to evaluate the clustering quality for different numbers of clusters.

**4(b) Clustering Using Minkowski Distance**

Minkowski distance (with p=3p=3p=3) was used for generalization beyond Euclidean distance:

- A distance matrix was computed, and Agglomerative Clustering was applied to the precomputed distances.
- Silhouette scores were computed similarly for a range of clusters to determine the optimal number of clusters.

**4(c) Clustering Using Cosine Similarity**

Cosine similarity measures the angle between vectors in high-dimensional space:

- Clustering was performed on a cosine similarity matrix, where data points with similar orientations were grouped together.
- Silhouette scores were again computed for cluster evaluation.

**5\. Observed Discrepancies Between Clustering Results**

When comparing the results of the clustering approaches using Euclidean distance, Minkowski distance, and Cosine similarity, some discrepancies were observed:

- **Euclidean Distance**: This measure clusters data based on physical proximity, which might not capture the relationships between data points in high-dimensional spaces like image data. The silhouette scores were lower compared to other measures, suggesting that the clusters were not as well-separated.
- **Minkowski Distance (p=3)**: As a generalization of Euclidean distance, the Minkowski distance introduced non-linearity into the clustering process. This helped identify more distinct clusters compared to Euclidean distance, with improved silhouette scores.
- **Cosine Similarity**: Cosine similarity performed the best in terms of silhouette scores. Since it measures the angle between vectors rather than their absolute distances, it is well-suited for high-dimensional datasets like facial images, where the relative orientation of data points is more significant than their magnitude. The clusters were more distinct, and this similarity measure captured more meaningful patterns in the data.

**6\. Choosing the Number of Clusters Using Silhouette Scores**

Silhouette scores were computed for each clustering approach across a range of clusters (2–40). The number of clusters that maximized the silhouette score was selected as the optimal number for each distance metric:

- **Euclidean Distance**: The optimal number of clusters was smaller, as the data points were grouped based on physical proximity, leading to more compact clusters.
- **Minkowski Distance**: A larger number of clusters was preferred, as it allowed for greater flexibility in identifying non-linear structures in the data.
- **Cosine Similarity**: The optimal number of clusters was higher than for Euclidean distance but still lower than Minkowski. The clusters were well-separated, reflecting the high dimensionality and the orientation-based similarity of the data.

**7\. Classifier Training Using the Clustered Data**

Once the optimal clusters were determined from the clustering process, the resulting clustered data (from Euclidean, Minkowski, or Cosine clustering) was used to train the SVM classifier again using k-fold cross-validation.

The results were as follows:

- **Euclidean Distance clusters**: Training on the Euclidean-based clustered data did not significantly improve the classifier's performance, likely due to the less distinct cluster boundaries.
- **Minkowski Distance clusters**: Training on these clusters yielded slightly better results due to the more flexible cluster definitions, but the improvement was marginal.
- **Cosine Similarity clusters**: The classifier performed the best when trained on data clustered using Cosine similarity. This was expected, as the clusters from Cosine similarity were better separated and reflected more meaningful relationships in the data.

**Conclusion**

In conclusion, the **Cosine similarity** clustering method produced the best results for both clustering quality (as measured by silhouette scores) and classification performance when training an SVM classifier. The discrepancies observed between the Euclidean, Minkowski, and Cosine similarity metrics were due to differences in how they measure distance and similarity, with Cosine similarity being the most appropriate for high-dimensional image data.
