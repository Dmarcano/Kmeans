#%% [markdown]
## K nearest neighbors algorithm
# The following code implements the k-means algorithm
#  for a set of given n-vectors belonging to the kth cluster denoted by x_i,
#         we try to find some representative z vectors z_k such that 

#         J_cluster_distance is as small as can be

#         for each vector x_i inside the cluster with representative vector z_k:
#             J_cluster_distance += 
            

#         J_cluster_distance = J_cluster_distance / len(# vectors in k)
#%%
import numpy as np
import random as rnd 
import matplotlib.pyplot as plt 

class KCluster(object):
    """
        Class that represents a single K means cluster grouping.
        Keeps track of:
            1. the clusters representative vector
            2. Its own vectors that belong to the cluster
            3. the grouping color vector
            4. The j_clust value 
    """
    def __init__(self, vector, Jclust = np.inf):
        self.rep_vector = vector # representative vector of vector cluster
        self.vec_list = [] # list of vector indices that map to ith vector that belongs to cluster
        self.colorVec = [] # color
        self.Jclust = Jclust
        self.Jclust_prev = 0
        self.c_vector = None
        self.optimized = False

    def clear_vec_list(self):
        self.vec_list = []

    def update_j_cluster(self):
    """
        Method that updates the J_cluster values 
    """ 
        self.Jclust_prev = self.Jclust

        # now update the J_cluster value

    def update_rep_vectors(self):
        pass



    def __repr__(self):
        return "K cluster:'rep vec: {}".format(str(self.rep_vector))

    def __str__(self):
        return "K Cluster: Rep vector: {}".format(str(self.rep_vector))

    


def k_means(k, vec_list ):
    """
        K means algorithm. Given k clusters and a list of n vectors 

        output:
            list of length k of the representative vectors for each cluster
    """

    """
        ========================= IMPLEMENTATION ==========================

        for a set of given n-vectors belonging to the kth cluster denoted by x_i,
        we try to find some representative z vectors z_k such that 

        J_cluster_distance is as small as can be

        for each vector x_i inside the cluster with representative vector z_k:
            J_cluster_distance += 
            
        J_cluster_distance = J_cluster_distance / len(# vectors in k)
        
        for each iteration:
            partition each vector to each group. aka find the 
    """
    # initialize k vectors
    kcluster_lists = choose_init_rep(k, vec_list)
    running = True

    if iterations != 0:
        running = False

    while running or iterations > 0:
        # iterate 
        update_vector_groups( vec_list, kcluster_lists)


    # keep track of variable J_clust and J_clust previous
    # J_clust_previous = 1
    # J_clust_curr = 0
    # keep iterating until J_clust == J_clust_previous:
    # while J_clust_previous is not J_clust_curr:

    #     # partition each vector in vec list  into the k group
    #     for vector in vec_list:
    #         # initialize min distance
    #         min_dist = np.abs(np.linalg.norm( vector, rep_vectors[0]))

    #         for rep_vector in rep_vectors:

    #             dist = np.abs(np.linalg.norm(vector, vec_list))
    #             min_dist = dist if dist < min_dist else min_dist


def choose_first_z(k, vec_list):

    """
        function which 
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
        # find the minimum 
        min_dist = np.inf
        min_k_cluster = cluster_list[0]
        

        for kcluster in cluster_list:
            dist =  np.linalg.norm( vector- kcluster.rep_vector ) # distance of np abs
            
            if dist < min_dist:
                min_k_cluster = kcluster
                min_dist = dist
        # now that we have reference to k_cluster that has minimum distance to current vector
        # add the idx to the current vector to the list of associated vectors to it
        # min_k_cluster.vec_list.append(idx)
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

def initialization_test():
    """
        Test on choosing the first z vectors.
         (testing choose_init_rep() )
    """
    k = 3
    vec_list = []
    for i in range(20):
        vec_list.append(np.array( [i, i + 1] ))

    vec_list = np.array(vec_list)
    
    for i in range(2):
        
        k_init_vecs = choose_first_z(k, vec_list)
        print(k_init_vecs)

     

def update_vector_groups_test():
    k = 2

    vec_list = []

    for i in range(10):
        vec = [i + 1 ,i + 1] if i % 2 else [-(i + 1), i + 1] # make a set of 2-vectors in q1 and q4
        np_vec = np.array( vec  ) # vectorize the vectors
        vec_list.append(np_vec)


    k_clusters = choose_first_z(k, vec_list) # get the kclusters

    update_vector_groups(vec_list, k_clusters)


    for idx, k_cluster in enumerate(k_clusters):
        print("==========. k-cluster: {} ==========".format(idx))
        
        for cluster_vec in k_cluster.vec_list:
            print(cluster_vec)
        print()


def kmeans_test():
    k = 2
    vec_list = []

    for i in range(10):
        vec = [i + 1 ,i + 1] if i % 2 else [-(i + 1), i + 1] # make a set of 2-vectors in q1 and q4
        np_vec = np.array(vec) # vectorize the vectors
        vec_list.append(np_vec)

    vec_list = np.array(vec_list)


    k_clusters = choose_first_z(k, vec_list) # get the kclusters

    update_vector_groups(vec_list, k_clusters) # creates two k clusters

    print(" ================ BEFORE ================ ")

    for cluster in k_clusters:
        print(cluster)

    update_rep_vectors(vec_list, k_clusters) # update rep vectors
    print(" ================ AFTER ================ ")

    for cluster in k_clusters:
        print(cluster)

kmeans_test()






#%%

def main():
    pass 
