from __future__ import division

import numpy as np
import time
import os
import sys
import h5py
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy.matlib
from sklearn import metrics

def extractTransientData(directory, filecount, features):
    """Extracts transient data from specified directory, usually 8 bit format.

        Parameters
        ----------
        directory : string,
            dir to the folder containing the transient data
            ie: /home/user/Transient_data
        filecount :int,
            Number of files in the directory
        features : int, 
            Number of features/variables/samples/x-axis
        Returns
        -------
        transient_data : array-like, shape =(filecount, features)
        """
    transient_data = np.empty((filecount, features), dtype = 'int8')
    for dirpath, dirnames, filenames in os.walk(directory):
        for n,f in enumerate(filenames):
            print "Found file:%s"%f
            transient_data[n] = np.fromfile(os.path.join(dirpath, f), dtype = 'int8')[0:features]
    print "Extracted Transient Data of shape:"
    print transient_data.shape
    print
    return transient_data

def extractSpectralData(directory, filecount, samples, features):
    """Extracts spectral data from specified directory, usually H5 file format with no groups just a single dataset.

        Parameters
        ----------
        directory : string,
         dir to the folder containing the transient data
            ie: /home/user/Spectral_data
        filecount :int,
             Number of files in the directory
        samples : int,
            Number of datapoints within the H5 dataset
        features : int,
             Number of features/variables/x-axis
        Returns
        -------
        transient_data : array-like, shape =(filecount, features)
        """
    Rows = filecount*samples
    Columns = features
    spectral_data = np.empty([Rows, Columns], dtype='float32')
    for dirpath, dirnames, filenames in os.walk(directory):
        for n,f in enumerate(filenames):
            print "Found file:%s"%f
            f = h5py.File("%s/%s"%(directory, f), "r")
            spectral_data[n*3600:(n+1)*3600] = np.array(f[f.keys()[0]],dtype='float32')
            f.close()
    print "Extracted Spectral Data of shape:"
    print spectral_data.shape
    print
    return spectral_data


def kpca(data, components=None, k="rbf", g=None, deg=9):
    """Performs Kernel-Principal Component Analysis on the specific data for dimensionality reduction and feature selection.
        Uses the Scikit-Learn KPCA Algorithm.

        Parameters
        ----------
        data :array-like, shape (samples, features),
            Input data for dimensionality reduction
        components : int, default = None,
            Number of components returned, if none are specified it is estimated
        k : string,
            Type of kernel to use ie: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        g : float, default = None
            Gamma value for specific kernel types, default is 1/(number of features)
        deg : int,
            Degree for polynomial kernels
        Returns
        -------
        KPCA_components : array-like, shape =(samples, components)
        """
    if(g == None):
        g = data.shape[1]
    
    print "\nKPCA Started, kernel = %s" %k
    print "Gamma = %e" %g
    start = time.time()
    K = KernelPCA(kernel=k,  gamma=g, degree=deg, remove_zero_eig=True, n_components = components)
    X = K.fit_transform(data)
    print "Time taken for KPCA: %.4f" %(time.time() - start)
    if(components == None):
        components = estimateComponents(K.lambdas_)
        print "Number of components not specified. Estimated components required = %d" % components
        return X[:,0:components]
    return X

def pca(data, components=None):
    """Performs Principal Component Analysis on the specific data for dimensionality reduction and feature selection.
        Uses the Scikit-Learn PCA Algorithm.

        Parameters
        ----------
        data :array-like, shape (samples, features),float64
            Input data for dimensionality reduction
        components : int, default = None,
            Number of components returned, if none are specified it is estimated
        Returns
        -------
        PCA_components : array-like, shape =(samples, components)
        """
    print "\nPCA Started"
    start = time.time()
    P = PCA(n_components = components)
    X = P.fit_transform(data)
    print "Time taken for PCA: %.4f" %(time.time() - start)

    if(components == None):
        components = estimateComponents(P.explained_variance_)
        print "Number of components not specified. Estimated components required = %d" % components
        return X[:,0:components]
    return X

def dbscan(data, minimum_samples=10):
    """Performs Density-Based Spatial Clustering of Applications with Noise on clustered data to produce labels.
        Estimates the min_samples paramater using two standard deviations of distance from the clusted data.
        Uses the Scikit-Learn DBSCAN Algorithm.

        Parameters
        ----------
        data :array-like, shape (samples, features),
            Input data for labelling
        minimum_samples : int, default = 10
            Minimum number of samples to be considered a cluster
        Returns
        -------
        labels : array-like, shape =[samples]
        """
    print "DBSCAN Started"
    start = time.time()
    sq_dists = pdist(data, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    eps = np.std(mat_sq_dists)*2
    if(eps <= 0):
        eps = 1e-6
    labels = DBSCAN(eps=eps, min_samples=minimum_samples, algorithm='auto').fit_predict(mat_sq_dists)
    print "Time taken for DBScan: %.4f" %(time.time() - start)
    print "Found labels: %r" % Counter(labels)
    return labels

def estimateGamma(data, start=1e-7, end=1e-6, tests=10, k="rbf", broad = True):
    """Performs a brute force iterative search for the optimum gamma value for KPCA based on maximum Silhouette score.
        Uses a linear space from start to end, its unwise to vary them by a large amount as this will cause more points to be generated at the larger value.
        If the specific start and end points are unknown, set broad to True which will do a broad search across a large space to narrow down new start and end points for the user.

        Parameters
        ----------
        data :array-like, shape (samples, features),
         Input data for KPCA
        start : float, default = 1e-7
            The first gamma value to be computed
        end : float, default = 1e-6
            The last gamma value to be computed
        tests : int, default =10,
            The number of iterations to be performed, determines the size of the linear space from start to end parameters
        k : string, default = "rbf"
         Type of kernel to use ie: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        broad : boolean, default = True,
            If True, uses a wide search path from 1e-18 to 100 in order to find a region with the best estimation. 
            Once this region is found the tests can be repeated with a more specific search path using the start and end parameters and broad = False.
        Returns
        -------
        maxgamma : float, optimised gamma value based on highest Silhouette score
        """
    print "Gamma estimation started for kernel: %s" %k

    if(broad):
        gammalist = 10.**np.arange(-18, 2)
        gammalist = np.hstack((gammalist, 1/data.shape[1]))
    else:
        gammalist = np.linspace(start, end, num=tests,endpoint=True)
    
    metriclist = np.empty(gammalist.shape[0])

    for n, i in enumerate(gammalist):
        X = kpca(data = data, k=k, g=i, components=50)
        labels = dbscan(X,10)
        labelcount = Counter(labels)
        if (len(labelcount) ==1):
            metriclist[n] = -1
        else:
            metriclist[n] = metrics.silhouette_score(X, labels, metric='euclidean')
        print "Silhouette score: %e\n" %metriclist[n]

    maxgamma = gammalist[np.argmax(metriclist)]
    print "Max gamma from silhouette score for kernel: %s = %e\n" % (k,maxgamma)

    return maxgamma

def estimateComponents(eigenvals):
    """Estimates the number of components required by finding the number of eigenvalues that account for 90% of the energy.
        If the estimated number of componets is less than 10 it returns 10. 
        Parameters
        ----------
        eigenvals : array-like, shape = [n_eigenvalues]
            The eigenvalues returned from the principal component anaylsis
        Returns
        -------
        n_components : int, 
        The number of components to be returned from the component analysis
        """
    if eigenvals.shape[0] == 0:
        return 10
    cumulative = np.cumsum(eigenvals)
    norm_cumulative = np.divide(cumulative, cumulative[-1])
    for n,i in enumerate(norm_cumulative):
        if(i >= 0.9):
            if (i < 10):
                return 10
            else:
                return n+1

def writeDataset(group, components, labels):
    """Creates the Components and Labels dataset within the specific H5 file group.

        Parameters
        ----------
        group : string,
            Name of the H5 group
        components : array-like, shape =(samples, features)
            The clustered, dimensionality reduced data
        labels : array-like, shape = [samples]
            The labels generated by DBSCAN from clustered data
        """
    group.create_dataset("Components", shape = components.shape, dtype='float32', data=components)
    group.create_dataset("Labels", shape = labels.shape, dtype='int16', data=labels)
    print "Data successfully written to H5 file\n"


if __name__ == "__main__":
    rawdata = extractSpectralData("./Raw_Spectra.h5", 6, 3600, 14200)
    outputH5 = h5py.File("./H5_Directory/SpectralData.h5", "w")

    # Create Group in output H5 file
    Kgrp = outputH5.create_group("KPCA")
    # Run KPCA on Raw Data from input H5 file
    Kernel = "rbf"
    gamma = estimateGamma(data = rawdata, k = Kernel)
    X = kpca(rawdata, g = gamma)
    labels = dbscan(X)
    writeDataset(group = Kgrp, components = X, labels = labels)

    # Create Group in output H5 file
    Siggrp = outputH5.create_group("Sigmoid")
    # Run KPCA on Raw Data from input H5 file
    Kernel = "sigmoid"
    gamma = estimateGamma(data = rawdata, k = Kernel)
    X = kpca(rawdata, g = gamma, k = Kernel)
    labels = dbscan(X)
    writeDataset(group = Siggrp, components = X, labels = labels)

    # Create Group in output H5 file
    Siggrp = outputH5.create_group("9th Poly")
    # Run KPCA on Raw Data from input H5 file
    Kernel = "poly"
    gamma = estimateGamma(data = rawdata, k = Kernel)
    X = kpca(rawdata, g = gamma, k = Kernel)
    labels = dbscan(X)
    writeDataset(group = Siggrp, components = X, labels = labels)

    # Create Group in output H5 file
    Pgrp = outputH5.create_group("PCA")
    #Run PCA and DBSCAN, save results in group
    X = pca(rawdata)
    Xlabels = dbscan(X, 10)
    writeDataset(group = Pgrp, components = X, labels = Xlabels)
    print "Data saved. Label Results:"
    print Counter(Xlabels)

    # Save Raw data as dataset not within a group
    outputH5.create_dataset("Rawdata", shape = rawdata.shape, dtype='float64', data=rawdata)

    inputH5.close()
    outputH5.close()