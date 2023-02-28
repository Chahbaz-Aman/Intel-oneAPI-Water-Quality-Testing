#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import daal4py as d4p
import joblib
import pandas as pd
import numpy as np
import sys


# In[ ]:


n = sys.argv[1]


# In[ ]:


d4p.daalinit() #initializes the distribution engine

infile = 'data/chunk{}.csv'.format(str(d4p.my_procid()))

# read data
X = pd.read_csv(infile)


# In[ ]:


# computing inital centroids
init_result = d4p.kmeans_init(nClusters = int(n), method = "plusPlusDense", distributed=True).compute(X)


# In[ ]:


# retrieving and printing inital centroids
centroids = init_result.centroids
print("Here's our centroids:\n\n\n", centroids, "\n")

centroids_filename = './models/centroids/kmeans_clustering_'+ n +'_clusters_initcentroids_'+  str(d4p.my_procid()+1) + '.csv'

# saving centroids to a file
joblib.dump(centroids, centroids_filename)


# In[ ]:


# loading the initial centroids from a file
loaded_centroids = joblib.load(open(centroids_filename, "rb"))
print("Here is our centroids loaded from file:\n\n",loaded_centroids)


# In[ ]:


# compute the clusters/centroids
kmeans_result = d4p.kmeans(nClusters = int(n), maxIterations = 5, assignFlag = True, accuracyThreshold = 1e-3, gamma=1).compute(X, init_result.centroids)


# In[ ]:


# retrieving and printing cluster assignments
assignments = kmeans_result.assignments

assignments_filename = './models/assignments/kmeans_clustering_'+ n +'_clusters_assignments_'+  str(d4p.my_procid()+1) + '.csv'

# saving centroids to a file
joblib.dump(assignments, assignments_filename)

print("Here is our cluster assignments for first 5 datapoints: \n\n", assignments[:5])

