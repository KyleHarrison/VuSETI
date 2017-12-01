from __future__ import division

import numpy	 as np
from scipy.spatial.distance import pdist, squareform

def sepIndex(X, t, k=5):
    """Calculates the separability index from input data and its labels. 
    The separability index is an indication of how many data points share their K-Nearest Neighbors class.
    A value of 0 represents poor clusters, as no data points share the majority of their neighbors label.abs
    A value of 1 represents ideal clustering where each class is completely separated from other classes.

        Parameters
        ----------
        X : array-like, shape = (samples, features),
            Input data, ideally clustered/dimensionality reduced features
        t : array-like, shape = features,
            The class labels associated with the each of the input array's samples
        k : int, default = 5
         Number of nearest neighbors, avoid using low values which would result in over-fitting
        Returns
        -------
        separability_index : float, value between 0 and 1
        """
    # Pairwise distance to each event, results in 1D array
    DistanceMx = pdist(X, 'sqeuclidean')
    # Form square array for distance to every point
    DistanceMx = squareform(DistanceMx)
    # Index of original sorted columns
    SortedIndex = np.argsort(DistanceMx)
    # Loop through the index of sorted columns for the 1st nearest neighbor
    total =0
    
    # If the specified K is higher than the total number of features it is set to that number.
    if(X.shape[1] < k):
        k = X.shape[1]

    for n, i in enumerate(SortedIndex):
        if(k==1): # KNN of 1
            if(t[i[1]] == t[n]): # If PC dist has the same label
                total += 1
        else: # KNN > 1
            sub_total = 0
            for x in range(1,k+1,1): # Loop through K
                if(t[i[x]] == t[n]):
                    sub_total += 1
            # if (sub_total/k == 0.5): # Round up from 0.5
            #     total +=1
            # else:
            #     print sub_total/k
            #     total += np.round(sub_total/k, 0)
            total += np.round(sub_total/k, 0)
    return float(total/X.shape[0])