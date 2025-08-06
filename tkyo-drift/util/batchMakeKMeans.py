# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
# This is good for vectors/matrices
import numpy as np
# This is the kmeans package
from sklearn.cluster import KMeans
# Allows the use of time functions
import time

def kMeansClustering(embeddings):

    # Starts the total function timer
    startTotal = time.perf_counter()

    # Sets num_vectors to the amount of vectors
    num_vectors = len(embeddings)
    

    # Sets dims to the length of the first vector in the array
    dims = len(embeddings[0])

    # Determines the amount of clusters
    num_of_clusters = int(np.sqrt(num_vectors / 2)*10)

    # Initialize the KMeans clustering algorithm with specific parameters:
    # random_state: Seed for random number generator to ensure reproducibility
    # (Using the same random_state will produce identical results across runs)
    print(f"Building KMeans for {num_vectors} vectors with {num_of_clusters} clusters.")
    kmeans = KMeans(num_of_clusters, random_state=42069)

    # Train the K-means model on our data
    # This is where the actual clustering algorithm runs:
    # 1. Initializes centroids (using k-means++ by default)
    # 2. Iteratively assigns points to nearest centroids
    # 3. Recomputes centroids as mean of assigned points
    # 4. Repeats until convergence or max iterations reached
    kmeans.fit(embeddings)

    # assigns the resulting array of centroids to centroids variable
    centroids = kmeans.cluster_centers_

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"KMeans cluster analysis completed in: {endTotal - startTotal:.2f} seconds")

    return centroids
