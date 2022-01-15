import config as cf
import numpy as np
from math import sqrt
from math import floor
from sklearn.cluster import *

# Maximum values
maximum_values = {
    "RGB": [255]
    , "HSV": [180 , 255 , 255 ]
}

# Minimum values
minimum_values = {
    "RGB": [0]
    , "HSV": [0 , 0 , 0]
}


# Modified maximum and minimum values
maximum_values_cluster = {
    "RGB": maximum_values.get("RGB")
    , "HSV": np.multiply(
        maximum_values.get("HSV")
        , [1 , cf.removal_interval.get("max")/100 , cf.removal_interval.get("max")/100]
    )
}
minimum_values_cluster = {
    "RGB": [minimum_values.get("RGB")]
    , "HSV": np.add(
        minimum_values.get("HSV")
        , [
            0
            , cf.removal_interval.get("min") * maximum_values.get("HSV")[1] / 100
            , cf.removal_interval.get("min") * maximum_values.get("HSV")[2] / 100
        ]
    )
}

# Maximum distance between elements (in terms of the current colour codification) - 50%
maximum_distance = np.multiply(
    0.5
    , sqrt(
        np.sum(
            ( maximum_values_cluster.get(cf.colour_format) - minimum_values_cluster.get(cf.colour_format) ) ** 2
        )
    )
)

# Define the dictionary of number of dimensions
dimensions = {"RGB": 3
              , "HSV": 3
              }

# White and black colours
white_colour = {
    "RGB": minimum_values.get("RGB") * dimensions.get("RGB")
    , "HSV": [ 0 , 0 , maximum_values.get("HSV")[2] ]
    , "limits": [ None , minimum_values_cluster.get(cf.colour_format)[1] , maximum_values.get(cf.colour_format)[2]*(1 - cf.removal_interval.get("min") / 100)]
}
black_colour = {
    "RGB": maximum_values.get("RGB") * dimensions.get("RGB")
    , "HSV": minimum_values.get("HSV")
    , "limits": [ None , None , minimum_values_cluster.get(cf.colour_format)[2] ]
}

# Define how to represent the pixels (projection, 3d)
projections = {"RGB": "3d"
               , "HSV": "polar"}

# DataFrame columns to export
dataFrameColumns = {
    "RGB": list(cf.colour_format)
    , "HSV": [list(cf.colour_format)[0]]
}

# Clustering functions
clustering_function = {
    'AffinityPropagation': AffinityPropagation(
            damping = 0.5
            , max_iter = cf.maximum_iterations
            , convergence_iter = floor( cf.maximum_iterations*0.5 ) # 50% of the iterations
            , copy = True
            , preference = None
            , affinity = 'euclidean'
            , verbose = False
            , random_state = None
        )
    , 'KMeans': KMeans(
        n_clusters = cf.number_colors
        , init = 'random' # 'k-means++'
        , n_init = 10
        , max_iter = cf.maximum_iterations # 300
        , tol = 1e-3
        , verbose = 0
        , random_state = None
        , copy_x = True
        , algorithm = 'auto'
    )
    , 'DBSCAN': DBSCAN(
        eps = 0.5
        , min_samples = 0.05 # At least 5% of the points
        , metric = 'euclidean'
        , metric_params = None
        , algorithm = 'auto' # Can be "auto", "ball_tree", "kd_tree", "brute"
        , leaf_size = 30
        , p = None
        , n_jobs = None
    )
    , 'AgglomerativeClustering': AgglomerativeClustering(
        n_clusters = None
        , affinity = 'euclidean'
        , memory = None
        , connectivity = None
        , compute_full_tree = 'auto'
        , linkage = 'ward'
        , distance_threshold = None
        , compute_distances = False
    )
    , 'Birch': Birch(
        threshold = 0.5
        , branching_factor = 50
        , n_clusters = None
        , compute_labels = True
        , copy = True
    )
    , 'SpectralClustering': SpectralClustering(
        n_clusters = cf.number_colors
        , eigen_solver = None
        , n_components = None
        , random_state = None
        , n_init = 10
        , gamma = 1.0
        , affinity = 'rbf'
        , n_neighbors = 10  # Ignored for affinity = "rbf"
        , eigen_tol = 0.0
        , assign_labels = 'kmeans'
        , degree = 3
        , coef0 = 1
        , kernel_params = None
        , n_jobs = None
        , verbose = False
    )
    , 'OPTICS': OPTICS(
        min_samples = 1
        , max_eps = maximum_distance
        , metric = 'euclidean'
        , p = 2 # (metric,p) = ('minkowski',2) implies euclidean
        , metric_params = None
        , cluster_method = 'xi' # or 'dbscan'
        , eps = maximum_distance
        , xi = 0.05
        , predecessor_correction = True
        , min_cluster_size = cf.pct_min_px_cluster / 100
        , algorithm = 'auto' # Can be {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
        , leaf_size = 30
        , memory = None
        , n_jobs = None
    )
}

# Colours exchangings
colour_exhanges = [
    [0 , 0]
    , [1 , 1]
    , [2 , 2]
    , [3 , 3]
    , [4 , 4]
    , [5 , 5]
]