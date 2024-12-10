import numpy as np
import matplotlib.pyplot as plt
import json

class Restaurant:
    def __init__(self, lat=0, long=0, rating=0, category="", name=""):
        self.lat = lat
        self.long = long
        self.rating = rating
        self.category = category
        self.name = name

restaurants = []

# Read file
with open('yelp_academic_dataset_business.json') as f:
    for line in f:
        entry = json.loads(line)
        if entry["city"] == "Santa Barbara":
            restaurant = Restaurant(
                lat=entry["latitude"],
                long=entry["longitude"],
                rating=entry["stars"],
                category=entry["attributes"],
                name=entry["name"]
            )
            restaurants.append(restaurant)

print("Number of businesses in Santa Barbara:", len(restaurants))

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Convert to NumPy array
locs = np.array([(restaurant.lat, restaurant.long) for restaurant in restaurants])

# K-means clustering function
def k_means_clustering(data, k, max_iters=100, tol=1e-4):
    """
    Custom K-means clustering algorithm.
    """
    np.random.seed(42)
    initial_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[initial_indices]

    for iteration in range(max_iters):
        # Assign clusters
        labels = np.array([np.argmin([euclidean_distance(point, centroid) for centroid in centroids]) for point in data])

        # Recompute centroids
        new_centroids = np.array([data[np.where(labels == cluster)].mean(axis=0) for cluster in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids

    return centroids, labels

# Run K-means clustering with 7 clusters
k = 7
centroids, labels = k_means_clustering(locs, k)

# Output centroids
print("Cluster Centroids:")
for i, c in enumerate(centroids):
    print(f"Cluster {i}: {c}")



# Defining the plotting helper function
def plotCurrent(X, Rnk, Kmus):
    N, D = np.shape(X)
    K = np.shape(Kmus)[0]

    InitColorMat = np.matrix([[1, 0, 0], 
                              [0, 1, 0],   
                              [0, 0, 1],
                              [0, 0, 0],
                              [1, 1, 0], 
                              [1, 0, 1], 
                              [0, 1, 1]])

    KColorMat = InitColorMat[0:K]
    colorVec = Rnk.dot(KColorMat)
    muColorVec = np.eye(K).dot(KColorMat)

    plt.scatter(X[:,0], X[:,1], edgecolors=colorVec, marker='o', facecolors='none', alpha=0.3)
    plt.scatter(Kmus[:,0], Kmus[:,1], c=muColorVec, marker='D', s=50)

def calcSqDistances(X, Kmus):
    dists = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Kmus.T) + np.sum(Kmus**2, axis=1)
    return dists

def determineRnk(sqDmat):
    closest_clusters = np.argmin(sqDmat, axis=1)
    rnk = np.eye(sqDmat.shape[1])[closest_clusters]
    return rnk

def recalcMus(X, Rnk):
    mus = np.dot(Rnk.T, X) / np.sum(Rnk, axis=0, keepdims=True).T
    return mus

def runKMeans(K, fileString):
    fig = plt.gcf()

    # Load data file specified by fileString (JSON format)
    with open(fileString, 'r') as f:
        data = json.load(f)
    
    # Extract longitude and latitude from JSON objects
    X = np.array([[obj['longitude'], obj['latitude']] for obj in data])

    # Determine and store data set information
    N, D = X.shape

    # Allocate space for the K mu vectors
    Kmus = np.zeros((K, D))

    # Initialize cluster centers by randomly picking points from the data
    rand_inds = np.random.permutation(N)
    Kmus = X[rand_inds[0:K],:]

    # Specify the maximum number of iterations to allow
    maxiters = 1000

    for iter in range(maxiters):
        # Calculate a squared distance matrix
        sqDmat = calcSqDistances(X, Kmus)

        # Determine the closest cluster center for each data vector
        Rnk = determineRnk(sqDmat)

        KmusOld = Kmus
        plotCurrent(X, Rnk, Kmus)
        plt.show()

        # Recalculate mu values based on cluster assignments
        Kmus = recalcMus(X, Rnk)

        # Check if the cluster centers have converged
        if sum(abs(KmusOld.flatten() - Kmus.flatten())) < 1e-6:
            break

    # Determine cluster assignments for each data point
    cluster_assignments = np.argmax(Rnk, axis=1)

    # Return cluster centers and assignments
    return Kmus, cluster_assignments
