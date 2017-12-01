import os
import sys
import h5py
import json
from flask import Flask,render_template, request, jsonify
from sklearn.decomposition import FastICA, PCA, KernelPCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from SI import sepIndex
import pickle
import numpy as np
import time

class numpyToPython(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(numpyToPython, self).default(obj)

app = Flask(__name__)
#ps -fA | grep python
@app.route('/')
def renderTemplate():
    databaselist = ListH5(directory)
    global X
    global labels
    global rawData
    return render_template('Webpage.html',
        initMetrics = json.dumps(metriclist, cls=numpyToPython), 
        initLabelcount = json.dumps(labelcount, cls=numpyToPython), 
        initData = json.dumps(X, cls=numpyToPython), 
        initRawData = json.dumps(rawData[0], cls=numpyToPython), 
        initLabels = json.dumps(labels, cls=numpyToPython))

@app.route('/getOrPostGroups', methods=['GET','POST'])
def getGroups():
    start = time.time()
    global groups
    if request.method == 'GET':
        return json.dumps(groups, cls=numpyToPython)
    else:
        db = request.get_json()
        f = h5py.File("%s/%s"%(directory,db), "r")
        keys = list(f.keys())
        groups = keys
        groups.remove("Rawdata")
        print "Groups load time: %.3f"%(time.time() - start)
        return json.dumps(groups, cls=numpyToPython)


@app.route('/postRawData', methods=['POST'])
def postRawData():
    index = request.get_json()
    return json.dumps(rawData[index], cls=numpyToPython)

@app.route('/postNewDataset', methods=['POST'])
def postNewDataset():
    start = time.time()
    global X
    global labels
    global rawData

    resp = request.get_json()
    db = resp['dbname']
    groupname = resp['groupname']

    f = h5py.File("%s/%s"%(directory,db), "r")
    G1 = f.get(groupname)

    X = np.asarray(G1.get("Components"))
    labels = np.asarray(G1.get("Labels"))
    rawData = np.asarray(f.get("Rawdata"))
    
    if isinstance(rawData, np.floating):
        print "Data is floating point, rounding to 5 decimal places"
        rawData = np.matrix.round(rawData, 5) 
        
    f.close()
    labelcount = []
    for key, value in Counter(labels).iteritems():
        labelcount.append([key,value])

    if (len(labelcount) ==1):
        metriclist = [-1, 0]
    else:
        metriclist = [metrics.silhouette_score(X, labels, metric='euclidean'), sepIndex(X,labels, 15)]

    X = np.matrix.round(X[:,0:10], 5)
        
    NewDataset = {'components': X, 'labels': labels, 'metrics': metriclist, 'labelcount': labelcount}
    print "New Dataset time: %.3f"%(time.time() - start)
    return json.dumps(NewDataset, cls=numpyToPython)


@app.route('/getDatabaseList', methods=['GET'])
def getDatabaseList():
    global databaselist
    return json.dumps(databaselist, cls=numpyToPython)


def ListH5(directory):
    files =[]
    for dirpath, dirnames, filenames in os.walk(directory):
        break
    return filenames


def Extract():
	files =[]
	for dirpath, dirnames, filenames in os.walk('/home/kyle/4022/H5_Files/'):
	    for f in filenames:
		files.append(os.path.join(dirpath, f))
	Data = []
	for f in files:
		if(f[-3:] != ".py"):
			f = h5py.File(f)
			X = np.array(f[f.keys()[0]])
			for i in X[:,0]:
				Data.append(i)
	return np.asarray(Data)


if __name__=="__main__":
    start = time.time()
    print "Server Init:"
    directory = "./H5_Directory"
    databaselist = ListH5(directory)
    print "Found: %i H5 Files for visualising" %len(databaselist)
    dbname = databaselist[0]
    print "First H5 file to be rendered on page load: %s" %dbname

    f = h5py.File("%s%s"%(directory,dbname), "r")
    keys = list(f.keys())
    groups = keys
    groups.remove("Rawdata")
    print "Found groups in the first H5 file: %s" %groups

    G1 = f.get(keys[0])
    X = np.asarray(G1.get("Components"))
    X = np.matrix.round(X[:,0:10], 5)

    labels = np.asarray(G1.get("Labels"))
    rawData = np.asarray(f.get("Rawdata"))
    if isinstance(rawData, np.floating):
        print "Data is floating point, rounding to 5 decimal places"
        rawData = np.matrix.round(rawData, 5) 

    metriclist = [metrics.silhouette_score(X, labels, metric='euclidean'), sepIndex(X, labels,15)]
    labelcount = []
    for key, value in Counter(labels).iteritems():
        labelcount.append([key,value])
    f.close()
    print "Server load time: %.3f"%(time.time() - start)
    app.run(debug = True,port=5001)
    
