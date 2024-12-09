import numpy as np
import matplotlib.pyplot as plt
import json

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

