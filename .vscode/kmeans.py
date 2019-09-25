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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import imageio

class KCluster(object):
    """
        Class that represents a single K means cluster grouping.
        Keeps track of:
            1. the clusters representative vector
            2. Its own vectors that belong to the cluster
            3. the grouping color vector
            4. The j_clust value 
    """
    
    def __init__(self, vector):
        self.rep_vector = vector # representative vector of vector cluster
        self.vec_list = [] # list of vector indices that map to ith vector that belongs to cluster
        self.colorVec = [] # color
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


def other_init_method(k, vec_list, min_dist = 2):
    """
    courtesy of:
        
        1. take the mean of the entire vector list. have that as the main center
        2. randomly pick vectors until their distance is at least min_dist
    """
    num_tries = 0
    clusterList = []

    if(type(vec_list)  == type(np.array([])) ):
        vec_arr = vec_list
    else:
        vec_arr = np.array(vec_list)

    center_vec = np.mean(vec_list, axis=0)
    clusterList.append(KCluster(center_vec))

    while len(clusterList < k):
        rand_vec = np.random.choice(vec_arr) # grab a random vector
        # check if distance of vectors is at least min_dist
        distance = np.linalg.norm(rand_vec - center_vec) 
        

def create_k_clusters(z_vecs):
    """
        function that returns a list of 
    """
    k_cluster_list = [KCluster(vector) for vector in z_vecs ]

    return k_cluster_list


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

    k_cluster_color = ['r', 'g', 'b', 'y', 'm', 'c']

    for i, k_idx in enumerate(cluster_assignments):
        # enumerate through 
        pic = plt.scatter(x_vals[i], y_vals[i], c = k_cluster_color[int(k_idx)], s = 20 )


def kmeans_image_arr(vec_list, x_dim, y_dim, k, cluster_assignments):

    x_vals = list( map(lambda num: num[x_dim], vec_list) ) 
    y_vals = list( map( lambda num: num[y_dim], vec_list) )  

    k_cluster_color = ['r', 'g', 'b', 'y', 'm', 'c']

    #k_cluster_color = np.array(k_cluster_color)
    fig = plt.figure(1)
    canvas = FigureCanvas(fig)

    for i, k_idx in enumerate(cluster_assignments):
        # enumerate through 
        pic = plt.scatter(x_vals[i], y_vals[i], c = k_cluster_color[int(k_idx)], s = 20 )

    fig.canvas.draw()   
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image



def image_arr_to_gif(image_arr):

    # imageio.mimsave('../movie.gif', image_arr,fps=3)

    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./powers.gif', image_arr, fps=5)



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
    k = 4
    vec_list = []

    for i in range(10):
        vec = [i + 1 ,i + 1, i +5 , i- 2] if i % 2 else [-(i + 1), i + 1, i +5 , i- 2] # make a set of 2-vectors in q1 and q4
        np_vec = np.array(vec) # vectorize the vectors
        vec_list.append(np_vec)

    for i in range(10):
        vec = [ -(i + 1), -(i + 1), i +5 , i- 2] if i % 2 else [(i + 1), -(i + 1), i +5 , i- 2] # make a set of 2-vectors in q1 and q4
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

    return rep_vectors, c_arr, J_clust




#%%[markdown]
## ========== K-Means Sample data =============
from sklearn.datasets.samples_generator import make_blobs


def k_means_10_iter_test():
    vec_list, vec_labels = make_blobs(n_samples= 10, centers=2, n_features=2, random_state=0) 
    k = 2
    J_clust_list = []
    rep_vectors, c_arr, J_clust = k_means(k, vec_list)

    J_clust_list.append(J_clust)

    visalize_kmeans(vec_list, 0, 1, k, c_arr)


    for i in range(10):
        k_cluster_list = create_k_clusters(rep_vectors)

        rep_vectors, c_arr, J_clust = k_means(k, vec_list, k_clusters=k_cluster_list)

    print(" ======================= MY K MEANS CLUSTER ============ ")
    plt.figure()
    visalize_kmeans(vec_list, 0, 1, k, c_arr)

    print(" ======================= ACTUAL CLUSTER ============ ")
    plt.figure()
    visalize_kmeans(vec_list, 0, 1, k, vec_labels)

k_means_10_iter_test()

#%%
def k_means_400_samples():
    pass


#%%
def k_means_mismatch_k():
    pass 


#%%
def k_means_10_iters_anim():
    vec_list, vec_labels = make_blobs(n_samples= 300, centers=3, n_features=2, random_state=0) 
    k = 3
    J_clust_list = []
    images_arr = []
    J_clust_prev = 0

    rep_vectors, c_arr, J_clust = k_means(k, vec_list)

    img = kmeans_image_arr(vec_list, 0, 1, k, c_arr)
    images_arr.append(img)

    J_clust_list.append(J_clust)

    for i in range(20):
        J_clust_prev = J_clust # set previous cluster val to current

        k_cluster_list = create_k_clusters(rep_vectors)
        rep_vectors, c_arr, J_clust = k_means(k, vec_list, k_clusters=k_cluster_list)
        img = kmeans_image_arr(vec_list, 0, 1, k, c_arr)
        images_arr.append(img)

        if J_clust_prev == J_clust:
            # if both clusters are the same
            print("local critical point found!")
            break;

    image_arr_to_gif(images_arr)

    print(" ======================= MY K MEANS CLUSTER ============ ")
    plt.figure()
    visalize_kmeans(vec_list, 0, 1, k, c_arr)

    print(" ======================= ACTUAL CLUSTER ============ ")
    plt.figure()
    visalize_kmeans(vec_list, 0, 1, k, vec_labels)



k_means_10_iters_anim()

#%%
# ================ REAL WORLD DATA =============


#%%[markdown]
## Results Analysis
# * Describe any different initial conditionsyou tried to analyze your data. What worked and what didn't?
# *  How many iterations did it take until the algorithm converged?
# * Can you extract any meaning from the clustering results?
# *  What value(s) of k are reasonable for your application and why?
# * Explain any intuition behind the final clustering result. For example, if you had chosenthe handwritten digits dataset shown in the text, you would analyze whether the clustering algorithm separatedeach digit into a different clusteror not. To figure this out,look at examples from your dataset, and how they were categorized.


#%%
