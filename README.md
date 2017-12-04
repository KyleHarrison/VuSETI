VuSETI
===========

Pronouced View SETI, this web server/client Python application can be used for visualisation of clustered data generated using machine learning algorithms from a separate script. This tool can help with data exploration and as a pre-processing tool for feature selection and dimensionality reduction in machine learning prior to training a classifier.
<p align="center">
<img src="https://i.imgur.com/MUZBnyf.png" width="80%">
</p>

Kyle Harrison: kyleharrison1994@gmail.com
Dr Amit Mishra: akmishra@ieee.org

Synopsis
--------
The rendered Bootstrap HTML page dynmically generates content based on input data and uses [Vis.JS](http://visjs.org/docs/graph3d/) for a 3D cluster visualisation scatter plot and [Canvas.JS](https://canvasjs.com/html5-javascript-line-chart/) for 1D raw data visualisation. Communication between the server and client is done in JSON via [AJAX jQuery](http://api.jquery.com/jquery.ajax/) and all event handlers are performed by jQuery.

The Data_Analyis_Script.py uses [Scikit-Learn](http://scikit-learn.org/stable/) for machine learning dimensionality reduction, feature selection and data clustering. The script uses component analysis techniques on unlabelled data stored in HDF5 files and clustering techniques to apply labels to the computed components. These algorithms are fundamental to unsupervised machine learning, using correlation and variance to derive class membership in order to pre-process datasets and make the process of training classifiers easier for other projects.

Dependencies:
--------
pip install -r requirements.txt

* Python 2.7
* NumPy
* SciPy
* Flask
* Scikit-Learn

# Description
This project formed the basis of my Computer and Electrical undergraduate thesis and was completed in a matter of weeks, as such there is much that can be added for improvement. If you would like to use this tool please contact me or post an issue. I have tried by best to document the tool in this readme, more can be found in my [report](https://drive.google.com/open?id=1pf6uNWS_O2K_zONyI8rN3c5c_2dm16Sr)

* HTML - the base design needs to be improved, it was built for functionality - not looks
* Server - Flask is a micro-framework and could be improved with a Django implementation
* Database - HDF5 files are great for storage and easy to use but the ideal system would have SQL database interaction
* Machine Learning - more analysis techniques can be added, better parameter estimation techniques are available


Machine Learning - Data analysis script
--------
The primary use case for this tool is analysis of Radio Frequency Interference data from time-domain transient and frequency-domain spectral data. By convention in astronomical data, the storage is done by HDF5 file.

The machine learning algorithms applied to raw data are:

* [KPCA](http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html) - Kernel Principal Component Analysis
* [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - Principal Component Analyis
* [DBSCAN](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) - Density-based spatial clustering of applications with noise

KPCA and PCA utilise the variance within unlabelled data generate clusters based on similarities and differences between samples. The components generated are new features for the data representing the greatest degrees of variances from the original features and linear combinations of those features.

DBSCAN is a density based clustering tool that applies labels to data in densily clustered space and labels outliers in areas of low density. 

KPCA's Gamma is estimated through an iterative brute force search. DBSCAN's eps is estimated through the standard deviation of euclidean pairwise distances.

Metrics for clustering used in parameter estimation are:
* [SI](https://pdfs.semanticscholar.org/dacc/c51597af27682033bcdae55097d3faea7864.pdf) - Thortons Separability Index 
* [Silhouette Score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

Analysis data is stored in H5 files with groups for anaylsis type and datasets within the groups for resulting components and labels from DBSCAN. The raw data used for analysis is recorded in a separate dataset. This format is shown below:
<p align="center">
<img src="https://i.imgur.com/PIoQzo4.png" width="20%">
</p>


System Outline
--------
<p align="center">
<img src="https://i.imgur.com/75ZoJLJ.png" width="65%">
</p>

The script for anaylsis is run separately to the server in order to prevent latency issues when run on slower systems. The server loads resulting H5 files requested by the client into memory and returns components, labels and raw data to the web page for visualisation.

A more detailed block diagram is shown below:
<p align="center">
<img src="https://i.imgur.com/u2z28Mo.png" width="75%">
</p>

Server Functionaliy
--------
The methods in Server.py using Flask to render HTML pages and H5Py to interact with HDF5 files is shown below:
<p align="center">
<img src="https://i.imgur.com/CDtCh95.png" width="65%">
</p>

Webpage Functionality
--------
The JavaScript interactions in Webpage.html are shown below:
<p align="center">
<img src="https://i.imgur.com/JRU9vGH.png" width="80%">
</p>


