#%% [markdown]
## K means algorithm
# The following code implements the k-means algorithm
#  for a set of given n-vectors belonging to the kth cluster denoted by x_i,
#         we try to find some representative z vectors z_k such that 

#         J_cluster_distance is as small as can be

#         for each vector x_i inside the cluster with representative vector z_k:
#             J_cluster_distance += 
            

#         J_cluster_distance = J_cluster_distance / len(# vectors in k)
#%% 
# =============================== K MEANS ALGORITHM LIBRARY CODE =============================
import numpy as np
import random as rnd 
import matplotlib.pyplot as plt 
import matplotlib.cm as mplcm
import matplotlib.colors as colors


class KCluster(object):
    """
        Class that represents a single K means cluster grouping.
        Keeps track of:
            1. the clusters representative vector
            2. Its own vectors that belong to the cluster
            3. the grouping color vector
            4. The j_clust value 
    """
    
    def __init__(self, vector, Jclust = 0):
        self.rep_vector = vector # representative vector of vector cluster
        self.vec_list = [] # list of vector indices that map to ith vector that belongs to cluster
        self.colorVec = [] # color
        self.Jclust = Jclust
        self.converged = False
        

    def clear_vec_list(self):
        self.vec_list = []

    def __repr__(self):
        return "K cluster:'rep vec: {}".format(str(self.rep_vector))

    def __str__(self):
        return "K Cluster: Rep vector: {}".format(str(self.rep_vector))

    


def k_means(k, vec_list, k_clusters = None ):
    """
        K means algorithm. Given k clusters and a list of n vectors 

        input:
            1. k - int for number of clusters
            2. vec_list - list/array of N vectors to classify under a cluster
            3. k_cluster [optional] - list/arr of k_cluster objects. defaults to first k 
            vectors to use for the k_clusters grouping

        output:
            1. list of length k of the representative vectors for each cluster
            2. c_arr of length N that classify each vector in vec_list to each cluster
            3. J cluster value for the iteration
    """

    vec_list_len = len(vec_list)
    c_arr = np.zeros(vec_list_len)

    if k_clusters == None:
        # initialize k_clusters
        k_cluster_list = choose_first_z(k, vec_list)
    else:
        k_cluster_list = k_clusters

    # associates vec_list vectors with specific cluster
    update_vector_groups(vec_list, k_cluster_list, c_arr) 
    update_rep_vectors(vec_list, k_cluster_list)

    # get the j_cluster
    J_cluster = calculate_Jcluster(k_cluster_list, vec_list)

    rep_vectors = []

    for k_cluster in k_cluster_list:

        rep_vec = k_cluster.rep_vector
        rep_vectors.append(rep_vec)
    
    return rep_vectors, c_arr, J_cluster

def calculate_Jcluster(k_cluster_list, vec_list):
    
    J_clust = 0
    N = len(vec_list)

    for k_cluster in k_cluster_list:
        #for each k_cluster find the individual cluster value J_i

        J_clust_indv = 0 # individual j cluster value
        for vector in k_cluster.vec_list:
            # for each vector in our list of vector take the norm squared and add it 
            # to the individual j cluster value
            norm_squared = np.linalg.norm(vector - k_cluster.rep_vector) ** 2
            J_clust_indv += norm_squared
        
        #divide J_i by N
        J_clust_indv /= N
        # add each individual J_i to the total J_cluster
        J_clust += J_clust_indv

    return J_clust


def choose_first_z(k, vec_list):

    """
        function to find initial k vectors which chooses the first k vectors in a given
        list as the initialization vectors for k means.

        input:
            k - number of clusters
            vec_list - list/array of N-dimensional numpy vectors

        output:
            list of KCluster Objects with their representative vectors 

        
    """
   
    k_cluster_vectors = [] # list of KCluster objects to return

    for i in range(k):
        vec_choice = vec_list[i]
       
        k_cluster_vectors.append(KCluster(vec_choice))

    return k_cluster_vectors
    

def update_vector_groups(vec_list, cluster_list, c_arr):
    """
        function that updates the groupings of a list/array of n-dimensional 
        numpy vectors and assigns them to their closest representative vector 
        given a list/array of k_cluster objects

        input:
            1. list/numpy_arr of n-dimensional numpy vectors
            2. list/numpy_arr of KCluster Objects
            3. list/numpy_arr of integer who's indeces
            correspond to the indeces of the first vector list and whos
            value corresponds to which KCluster they belong to.

        output:
            1. the c_arr will be mutated so that it's values correspond to 
            the closest kCluster
            2. each individual KCluster will have its member vector list mutated with the vectors
            closest to the representative vector 
    """

    
    # clear all the kcluster lists
    for kcluster in cluster_list:
        kcluster.clear_vec_list()

    for idx, vector in enumerate(vec_list):
        # find the minimum distance between one vector and all the rep_vectors
        min_dist = np.inf
        min_k_cluster = cluster_list[0]
        min_k_idx = 0
        

        for k_idx, kcluster in enumerate(cluster_list):
            dist =  np.linalg.norm( vector- kcluster.rep_vector ) # distance of np abs
            
            if dist < min_dist:
                min_k_cluster = kcluster
                min_dist = dist
                min_k_idx = k_idx
        # now that we have reference to k_cluster that has minimum distance 
        # to current vector, we associate the vector the the kcluster and update the c_arr
        # assignment 
        
        c_arr[idx] = min_k_idx
        min_k_cluster.vec_list.append(vector)

    return 

def update_rep_vectors(vec_list, k_cluster_list):
    """
        updates all the representative kCluster representative vecotrs given two lists or numpy arrays of:
            1. numpy-n-vectors 
            2. kCluster objects
    """ 

    vec_len = np.size( vec_list[0] )


    for k_cluster in k_cluster_list:

        average_vec = np.mean(k_cluster.vec_list, axis = 0)
        #print(average_vec)
        k_cluster.rep_vector = average_vec



        
    # for each cluster have to take average of all the vectors 



#%%
# =============================== K MEANS ALGORITHM UNIT TESTING =============================

# K means unit testing. No rigorous testing just incremental tests with exprected Input
# Outputs

def initialization_test():
    """
        Test on choosing the first z vectors.
         (testing choose_init_rep() )

         Print some i vectors to see that the initialization method does indeed
         choose the first k vectors as initialization z vectors
    """
    k = 3
    vec_list = []
    for i in range(20):
        vec_list.append(np.array( [i, i + 1] ))

    vec_list = np.array(vec_list)
    print(vec_list)

    for i in range(2):
        
        k_init_vecs = choose_first_z(k, vec_list)
        print(k_init_vecs)
    

def update_vector_groups_test():
    """
        Updating the vector  groups test
        Create a set of vectors in 2-D space with one set around Q1 and 
        other in Q4
    """

    k = 2 # two k groups

    vec_list = []

    c_arr = np.zeros(10)

    for i in range(10):
        # all odd vectors are in q4 and all even vectors are in q1
        vec = [i + 1 ,i + 1] if i % 2 else [-(i + 1), i + 1] 
        np_vec = np.array( vec  ) # vectorize the vectors
        vec_list.append(np_vec)


    k_clusters = choose_first_z(k, vec_list) # get the kclusters

    update_vector_groups(vec_list, k_clusters, c_arr)


    for idx, k_cluster in enumerate(k_clusters):
        print("==========. k-cluster: {} ==========".format(idx))
        
        for cluster_vec in k_cluster.vec_list:
            print(cluster_vec)
        print()

    print(c_arr)

def kmeans_test_parts():

    """
        test that tests a full hard coded k means run using all the parts for k means
    """
    
    k = 2
    vec_list = []

    c_arr = np.zeros(10)

    for i in range(10):
        vec = [i + 1 ,i + 1] if i % 2 else [-(i + 1), i + 1] # make a set of 2-vectors in q1 and q4
        np_vec = np.array(vec) # vectorize the vectors
        vec_list.append(np_vec)

    vec_list = np.array(vec_list)


    k_clusters = choose_first_z(k, vec_list) # get the kclusters

    update_vector_groups(vec_list, k_clusters, c_arr) # creates two k clusters

    print(" ================ BEFORE ================ ")

    for cluster in k_clusters:
        print(cluster)

    update_rep_vectors(vec_list, k_clusters) # update rep vectors
    print(" ================ AFTER ================ ")

    for cluster in k_clusters:
        print(cluster)

def k_means_1_iter_test():
    """
        Creating sample 2-D data and calling a k means algorithm for one iteration

        Testing that we get results we expect
    """
    k = 2
    vec_list = []

    for i in range(10):
        vec = [i + 1 ,i + 1] if i % 2 else [-(i + 1), i + 1] # make a set of 2-vectors in q1 and q4
        np_vec = np.array(vec) # vectorize the vectors
        vec_list.append(np_vec)


    rep_vectors, c_arr, J_clust = k_means(k, vec_list)


    visalize_kmeans(vec_list, 0, 1, k, c_arr)
    # print("===================== REP VECTORS =====================")
    # print(rep_vectors)
    # print()
    # print("===================== C ARRAY =====================")
    # print(c_arr)
    # print()
    # print("===================== J CLUST =====================")
    # print(J_clust)
    # print()


k_means_1_iter_test()


#%%
# =============================== K MEANS VISUALIZATION =============================

def visalize_kmeans(vec_list, x_dim, y_dim, k, cluster_assignments ):
    """
        Function that visualizes a projection of the outputs of a kmeans algorithm 
        across two specified dimensions

        inputs: 
            1. vec_list - list/array of N-dimensional vectors
            2. x_dim - vector dimension to use for x-axis
            3. y_dim - vector dimension to use for the y-axis
            4. k - number of clusters in the group
            5. cluster_assignments - vector whose indices correspond 
            to the indices of vec_list
    """
    # use map to map lambda that returns the [ith] element of a vector
    # to get the dimension vectors
    x_vals = list( map(lambda num: num[x_dim], vec_list) ) 
    y_vals = list( map( lambda num: num[y_dim], vec_list) )  



    # start by assigning k types of colors.
    color_map = plt.get_cmap("gist_rainbow")
    cNorm  = colors.Normalize(vmin=0, vmax=k-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)



#%%
