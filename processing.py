import numpy as np
import math
import cv2
from PIL import ImageColor
from skimage.transform import downscale_local_mean
from sklearn.cluster import *
import config as cf
import constants as c
import functions as f
import conversions as convers

# Read image
def read_image(files , format):

    image = cv2.imread( files )[:, :, ::-1].astype(np.uint8)

    return convers.image_conversion(image , "RGB" , cf.colour_format )

    #return {
    #   "RGB": cv2.imread( files )[:, :, ::-1].astype(np.uint8)
    #    , "HSV": cv2.cvtColor( cv2.imread( files ) , cv2.COLOR_BGR2HSV )
    #}.get(format)


# Extract all points from image into a nx3 vector
def extract_pixels(image, factor):

    # Change image format to RGB
    #image = convers.change_image_format(image, cf.colour_format, "RGB")

    # Image resampling
    array = downscale_local_mean(image
                                 , factors = (factor , factor , 1)
                                 )

    # Array extraction
    array = np.reshape(array,
                       (array.shape[0] * array.shape[1]
                        , array.shape[2]
                        )
                       )

    return array


# Histogram solution (extract N most important colours)
def histogram(array):

    # Assign variables
    maximum_values = c.maximum_values[cf.colour_format] if len(c.maximum_values[cf.colour_format]) == 3 else c.maximum_values[cf.colour_format][0]*np.ones(3, dtype = np.int8)
    number_bins = cf.number_bins[cf.colour_format] if len(cf.number_bins[cf.colour_format]) == 3 else cf.number_bins[cf.colour_format][0]*np.ones(3, dtype = np.int8)

    # If array has 3D, turn into 2D
    if (len(array.shape) == 3):
        array = np.reshape(array,
                           (array.shape[0] * array.shape[1],
                            array.shape[2]))

    # Declare variables
    module = np.divide(maximum_values, number_bins)     # Division of the 3D space
    hist2gram = np.zeros( np.append(number_bins,4) )       # Add the fourth dimension

    # For each element in the array, let's get to which region it belongs
    for a in array:

        # If maximum value, then assign 2. If not, get the index of the region in each dimension
        c1 = int(2 if a[0] == maximum_values[0] else np.floor_divide(a[0], module[0]))
        c2 = int(2 if a[1] == maximum_values[1] else np.floor_divide(a[1], module[1]))
        c3 = int(2 if a[2] == maximum_values[2] else np.floor_divide(a[2], module[2]))

        # Add 1 to the count and assign colour to region
        hist2gram[c1, c2, c3, 0] = hist2gram[c1, c2, c3, 0] + 1
        hist2gram[c1, c2, c3, 1] = (c1 + 0.5) * module[0]
        hist2gram[c1, c2, c3, 2] = (c2 + 0.5) * module[1]
        hist2gram[c1, c2, c3, 3] = (c3 + 0.5) * module[2]

    # Reshape elements
    hist2gram = np.reshape(hist2gram,
                           [np.prod(number_bins), 4])

    # Order (most appearances first)
    hist2gram = hist2gram[np.argsort(hist2gram[:, 0])][::-1]

    # Get number_colors first appearances (1: to remove the count of associated pixels)
    hist2gram = hist2gram[range(cf.number_colors), 1:]

    return hist2gram


def palette_change(image, originPalette, destinyPalette):

    # Downsize image
    image = downscale_local_mean(image,
                                 factors = (cf._sampling_rate, cf._sampling_rate, 1))

    # If image has 3D, turn into 2D
    dimensions = image.shape
    if (len(image.shape) == 3):
        image = np.reshape(image,
                           (image.shape[0] * image.shape[1],
                            image.shape[2]))

    # Get transform ratios
    ratio = []
    palette_names = []
    new_palette = []
    for iter in range(originPalette.shape[0]):

        # Compose ratios
        color_new_palette = ImageColor.getcolor( destinyPalette[iter] , cf.colour_format )
        new_palette = np.append( new_palette , color_new_palette)
        ratio = np.append( ratio , color_new_palette - originPalette[iter,:])

        # Compose names of the palettes
        palette_names = np.append(palette_names, ''.join(str(originPalette[iter,:])))

    # Reshapes
    ratio = np.reshape(ratio, originPalette.shape)
    new_palette = np.reshape(new_palette, originPalette.shape)

    # For each element in the image, perform transformation based on ratio, depending on the colour it is assigned to
    for iter in range(image.shape[0]):

        # Find the nearest point for current image
        near_point = f.nearest_point(image[iter,:], originPalette)

        # Find the palette element and apply corresponding ratio
        index = np.where(palette_names == ''.join(str(near_point)))

        # Apply ratio to the current element
        image[iter,:] = image[iter,:] + ratio[index,:]

    # Establish boundaries of the image
    image = normalizeImageFunction( image , cf.colour_format )

    # Cast the whole image to int, and resize
    image = image.astype(np.uint8)
    image = np.reshape(image, dimensions)

    return image, new_palette


def cluster(array, method):

    # If array has 3D, turn into 2D
    if (len(array.shape) == 3):
        array = np.reshape(array,
                           (array.shape[0] * array.shape[1],
                            array.shape[2]))

    # Remove incorrect pixels
    array = removeIncorrectPx(
        array
        , cf.colour_format
    )

    # Apply one clustering technique
    clustering = {
        'AffinityPropagation': AffinityPropagation(
                damping = 0.5
                , max_iter = cf.maximum_iterations
                , convergence_iter = math.floor( cf.maximum_iterations*0.5 ) # 50% of the iterations
                , copy = True
                , preference = None
                , affinity = 'euclidean'
                , verbose = False
                , random_state = None
            )
        , 'KMeans': KMeans(
            n_clusters = cf.number_colors # math.floor( 2*cf.number_colors ) # At least a 50% more of the number of colours to be extracted
            , init = 'k-means++'
            , n_init = 10
            , max_iter = cf.maximum_iterations
            , tol = 1e-4
            , verbose = 0
            , random_state = 0
            , copy_x = True
            , algorithm = 'auto'
        )
        , 'DBSCAN': DBSCAN(
            eps = 0.5
            , min_samples = math.floor( max( array.shape ) * 0.05 ) # At least 5% of the points
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
            n_clusters = math.floor( 1.5*cf.number_colors ) # At least a 50% more of the number of colours to be extracted
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
    }.get( method ).fit( array )

    # Extract the clustering centers, and the label of each points
    labels = clustering.labels_

    # If all the labels are -1, the clustering algorithm has not worked. Use as a backup KMeans
    if all(labels == -1):
        print(method+" has not provided any results. Instead, KMeans method is used instead")
        return cluster(array, "KMeans")

    # Get all the centroids
    centroids = getCentroids(array, labels)

    # Compose the count of the clusters
    centroid_order = np.zeros((max(centroids.shape), 2))
    for r in range(len(centroids)):
        centroid_order[r, 0] = int(r)
        centroid_order[r, 1] = len(labels[labels == r])

    # Reorder centroids
    centroid_order = centroid_order[np.argsort(centroid_order[:, 1])][::-1]

    # Assign the N first elements to the palette
    palette = centroids[
              centroid_order[:cf.number_colors, 0].astype(np.uint8), :
    ].astype(np.uint8)

    return palette


def getCentroids(array, labels):

    # Once the clustering has been carried out, get the centroids based on the points and their labelling
    centroids = np.zeros( (
        max(labels)+1
        , c.dimensions.get(cf.colour_format)
    ) )

    # Compute for all the elements in the array, the mean coordinates
    for r in range(0,max(centroids.shape)):
        centroids[r,:] = np.mean(
            array[labels == r]
            , axis = 0
        )

    return centroids


def normalizeImageFunction(image, format):

    # Get maximum and minimum values
    minimum_values = c.minimum_values.get(format) if len(c.minimum_values.get(format)) == c.dimensions.get(format) else c.minimum_values.get(format)[0]*np.ones(c.dimensions.get(format))
    maximum_values = c.maximum_values.get(format) if len(c.maximum_values.get(format)) == c.dimensions.get(format) else c.maximum_values.get(format)[0]*np.ones(c.dimensions.get(format))

    # Normalize image (establish boundaries)
    boundariesFunction = lambda x, y, z : np.clip( x , y , z)
    normalizeFunction = lambda x : np.array(
        [boundariesFunction( x[i] , minimum_values[i] , maximum_values[i] ) for i in range(len(x))]
    )

    # If the image is a 3D array
    if len(image.shape) > 1:
        image = np.array(
            [ normalizeFunction(i) for i in image ]
        )
    else:
        image = normalizeFunction(image)

    return image


def removeIncorrectPx(array , format):

    # If array has 3D, turn into 2D
    if (len(array.shape) == 3):
        array = np.reshape(array,
                           (array.shape[0] * array.shape[1],
                            array.shape[2]))

    # For every pixel, check whether it is valid or not
    if len(array.shape) > 1:
        valid_values = np.array(
            [ f.isInRange( a , c.minimum_values_cluster.get(format) , c.maximum_values_cluster.get(format) ) for a in array ]
        )
    else:
        valid_values = f.isInRange( array , c.minimum_values_cluster.get(format) , c.maximum_values_cluster.get(format) )

    return array[valid_values == True , :]