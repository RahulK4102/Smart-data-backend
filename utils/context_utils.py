import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest

def detect_optimal_clusters(scaled_features):
    num_features = scaled_features.shape[0]
    print(f"Number of features: {num_features}")

    sse = []
    silhouette_scores = []
    cluster_range = range(2, min(11, num_features))

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

    # Plotting the Elbow method results
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(cluster_range, sse, marker='o')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('SSE')

    # plt.subplot(1, 2, 2)
    # plt.plot(cluster_range, silhouette_scores, marker='o')
    # plt.title('Silhouette Scores')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')

    # plt.tight_layout()
    # plt.show()

    # Using Kneedle to find elbow point
    kneedle = KneeLocator(cluster_range, sse, curve='convex', direction='decreasing')
    elbow_point = kneedle.elbow
    # print(f"Elbow point determined: {elbow_point}")
    
    if(elbow_point!=None):
        optimal_k = elbow_point
        if optimal_k > 1:  # Check that we have a valid elbow point
            # Compare with silhouette scores
            silhouette_optimal_k = cluster_range[np.argmax(silhouette_scores)]
            if silhouette_scores[silhouette_optimal_k - 2] > silhouette_scores[optimal_k - 2]:
                optimal_k = silhouette_optimal_k
    else:
        if(num_features<3):
            return num_features-1
        optimal_k=num_features-2

    # print(f"Optimal number of clusters selected: {optimal_k}")
    return optimal_k


def extract_datetime_features(df, col):
    """Extract features from datetime columns."""
    date_col = pd.to_datetime(df[col], errors='coerce')
    return [
        date_col.min().timestamp() if not date_col.min() is pd.NaT else 0,  # min as timestamp
        date_col.max().timestamp() if not date_col.max() is pd.NaT else 0,  # max as timestamp
        (date_col.max() - date_col.min()).days if date_col.min() is not pd.NaT else 0,  # duration in days
        date_col.dt.year.nunique(),
        date_col.dt.month.nunique(),
        date_col.dt.day.nunique(),
        date_col.isnull().mean()
    ]

def detect_context_with_clustering(df, report_path):
    column_features = []
    columns = df.columns
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:  # Numerical columns
            column_features.append([
                df[col].mean(),
                df[col].std(),
                df[col].min(),
                df[col].max(),
                df[col].isnull().mean(),
                0,  # Padding for categorical and datetime features
                0   # Padding for categorical and datetime features
            ])
        elif df[col].dtype == 'object':  # Categorical columns
            column_features.append([
                df[col].nunique(),  # Number of unique values
                df[col].isnull().mean(),  # Proportion of missing values
                len(df[col].unique()) / len(df),  # Ratio of unique values to total rows
                df[col].apply(lambda x: len(str(x)) if pd.notnull(x) else 0).mean(),  # Avg string length
                0,  # Padding for datetime features
                0,  # Padding for datetime features
                0   # Padding for datetime features
            ])
        elif np.issubdtype(df[col].dtype, np.datetime64):  # Datetime columns
            column_features.append(extract_datetime_features(df, col))
        else:
            # Fallback for unsupported column types, provide default values
            column_features.append([0] * 7)  # Padding to ensure consistent length
    
    # Convert the list of lists to a numpy array and handle potential inconsistencies
    try:
        column_features = np.array(column_features, dtype=float)  # Ensuring all elements are float
    except ValueError as e:
        print(f"Error converting column features: {e}")
        return

    # Normalize the features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(column_features.tolist())  # Convert to list before scaling to handle dtype

    # Determine the optimal number of clusters
    optimal_k = detect_optimal_clusters(scaled_features)
    
    # K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    column_clusters = kmeans.fit_predict(scaled_features)
    
    # Organize and write context detection results to the report
    cluster_contexts = {i: [] for i in range(optimal_k)}
    for i, col in enumerate(columns):
        cluster_contexts[column_clusters[i]].append(col)
    
    # Append context results to the report
    with open(report_path, 'a') as f:
        f.write("\nContext Detection Using Clustering:\n")
        for cluster, cols in cluster_contexts.items():
            f.write(f"- Cluster {cluster} contains columns: {cols}\n")
    
    print(f"Context detection results appended to {report_path}")