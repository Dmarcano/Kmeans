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
    def __init__(self, vector):
        self.rep_vector = vector # representative vector of vector cluster
        self.vec_list = [] # list of vector indices that map to ith vector that belongs to cluster
        self.colorVec = [] # color
        self.Jclust = np.inf
        self.Jclust_prev = 0
        self.c_vector = None

    def clear_vec_list(self):
        self.vec_list = []

    def update_j_cluster(self):
        pass 

    


def k_means(k, vec_list, iterations = None ):
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

    while running:
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


        

def choose_init_rep(k, vec_list):
    rep_vectors = [] # list to keep track of vectors used

    k_cluster_vectors = [] # list of KCluster objects to return

    # choose random vectors from given list of vectors as rep vectors
    while len(rep_vectors) is not k:
        vec_choice = rnd.choice(vec_list)
        if vec_choice not in rep_vectors:
            rep_vectors.append(vec_choice)
            k_cluster_vectors.append(KCluster(vec_choice))

    return k_cluster_vectors

    

def update_vector_groups(vec_list, cluster_list):
    #update the group of vectors given a list of vectors and k vectors

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
        print(average_vec)
        k_cluster.rep_vector = average_vec



        
    # for each cluster have to take average of all the vectors 



#%%

def initialization_test():
    pass 


def update_vector_groups():
    pass 