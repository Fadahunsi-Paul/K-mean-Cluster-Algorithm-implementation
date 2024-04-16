import numpy as np

def k_means(X, k, max_iters=100): 
    try:
        # Step 1: Randomly initialize centroids
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]


        for _ in range(max_iters): 

            # Step 2: Assign data points to nearest centroid
            distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
            labels = np.argmin(distances, axis=0)
            
            # Step 3: Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids, labels 
    except Exception as e:
        print("An error occurred:", e)

# EXAMPLE IS GIVEN BELOW
try:
    X = np.array([[1, 2], [7, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
    k = 3

    #Finding the Centroids
    centroids, labels = k_means(X, k) 
    print("Final centroids:\n", centroids)
    print("Labels:", labels)
    print("""
    
    """)
    #finding eucledian distance
    distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
    print("Final Distance:\n", distances)
    print("""
    
    """)
    #Assigning the labels to the dataset
    for i in range(k):
        cluster_i_indices = np.where(labels == i)[0]
        cluster_i_points = X[cluster_i_indices]
        print(f"Data points in cluster {i}:")
        print(cluster_i_points)
        print("""
    
    """)
    #Calculating for new Centeroid
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    print("New Centroid:\n",new_centroids) 

except Exception as e:
        print("An error occurred:", e)