import numpy as np
import pandas as pd
import copy
import math
import cv2
from PIL import ImageColor
from skimage.transform import downscale_local_mean
import config as cf
import constants as c
import functions as f
import visualization as v
import conversions as convers

# Read image
def read_image(files , format):

    image = cv2.imread( files )[:, :, ::-1].astype(np.uint8)

    return convers.image_conversion(image , "RGB" , cf.colour_format )


# Extract all points from image into a nx3 vector
def extract_pixels(image, factor):

    # Image resampling
    array = downscale_local_mean(image
                                 , factors = (factor , factor , 1)
                                 )

    # Change the size of the array
    array = np.reshape(array,
                       (np.prod(array.shape[0:len(array.shape) - 1]),
                        array.shape[len(array.shape) - 1])
                       )

    return array


def palette_extracting( array , config ):

    # Palette extracting function
    paletteFunction = {
        "histogram": lambda x,y: histogram(x)
        , "clustering": lambda x,y: cluster(x , y.get("mode") )
    }.get(
        config.get("technique")
    )

    return paletteFunction( array , config )


# Histogram solution (extract N most important colours)
def histogram(array):

    # Assign variables
    maximum_values = c.maximum_values[cf.colour_format] if len(c.maximum_values[cf.colour_format]) == 3 else c.maximum_values[cf.colour_format][0]*np.ones(3, dtype = np.int8)
    number_bins = cf.number_bins[cf.colour_format] if len(cf.number_bins[cf.colour_format]) == 3 else cf.number_bins[cf.colour_format][0]*np.ones(3, dtype = np.int8)

    # Change the size of the array (if necessary)
    if (len(array.shape) == c.dimensions.get(cf.colour_format)):
        array = np.reshape(array,
                           (np.prod(array.shape[0:len(array.shape)-1]),
                            array.shape[len(array.shape)-1])
                           )

    # Correct extreme values (white-black) if necessary
    array = extremeColoursCorrection(
        array
        , cf.colour_format
    )

    # Declare variables
    module = np.divide(maximum_values, number_bins)        # Division of the 3D-space
    dimensions = np.append(
        number_bins
        , array.shape[len(array.shape)-1] + 2
    )
    hist2gram = np.zeros(dimensions)                       # Add N+2 dimensions (counting, the N-colours vector and the label)

    # For each element in the array, assign the region whom it belongs
    labels = []
    for a in array:

        # If maximum value, then assign N-1. If not, get the index of the region in each dimension
        c0 = [ int(number_bins[r]-1 if a[r] == maximum_values[r] else np.floor_divide(a[r], module[r])) for r in range(0,len(a)) ]
        centroid = copy.copy(c0)

        # Counting index and centroid of the cluster
        counting_index = tuple(
            np.append( c0 , 0 )
        )
        centroid.append( range(1 , len(a)+1) )
        label = tuple(
            np.append( c0 , len(a)+1 )
        )

        # Add 1 to the count and assign the centroid colour
        hist2gram[counting_index] = hist2gram[counting_index] + 1
        hist2gram[centroid] = np.multiply(
            np.add(c0 , 0.5)    # 0.5 as it is the center of the region
            , module
        )
        hist2gram[label] = int(''.join(list([str(c) for c in c0])))

        # Label that array element (c0 as string)
        labels.append( hist2gram[label] )

    # Reshape elements
    hist2gram = np.reshape(hist2gram,
                           [np.prod(number_bins), dimensions[::-1][0]])

    # Order (most appearances first)
    hist2gram = hist2gram[
                    np.argsort(hist2gram[:, 0]),:
                ][::-1]

    # Get number_colors first appearances (1: to remove the count of associated pixels)
    hist2gram = hist2gram[range(cf.number_colors), 1:]

    # Compose the output dictionary
    output = {}
    output["palette"] = hist2gram[ range(cf.number_colors) , : ][ : , range(1 , len(a)+1) ]
    output["palette_labels"] = { h[len(h)-1]: h[range(1,len(h))] for h in hist2gram }
    output["labels"] = labels
    output["array"] = array

    return output


def palette_change(image, originPalette, destinyPalette):

    # Downsize image
    image = downscale_local_mean(image,
                                 factors = (cf._sampling_rate, cf._sampling_rate, 1))

    # If image has 3D, turn into 2D
    dimensions = image.shape
    # Change the size of the array (if necessary)
    if (len(image.shape) == c.dimensions.get(cf.colour_format)):
        array = np.reshape(image,
                           (np.prod(image.shape[0:len(image.shape) - 1]),
                            image.shape[len(image.shape) - 1])
                           )

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

    # Change the size of the array (if necessary)
    if (len(array.shape) == c.dimensions.get(cf.colour_format)):
        array = np.reshape(array,
                           (np.prod(array.shape[0:len(array.shape) - 1]),
                            array.shape[len(array.shape) - 1])
                           )

    # Correct extreme values (white-black) if necessary
    array = extremeColoursCorrection(
        array
        , cf.colour_format
    )

    # Extract points and generate dataFrame
    points = pd.DataFrame(
        columns = list(cf.colour_format)
    )
    for col in points.columns:
        points[col] = array[ : , cf.colour_format.find(col) ]
    points["Aux"] = 0

    # Apply one clustering technique
    clusteringFunction = c.clustering_function.get( method )

    # Extract the clustering centers, and the label of each points
    clustering = clusteringFunction.fit( points[c.dataFrameColumns.get(cf.colour_format)] )
    labels = clustering.labels_

    # If there is less than Number Colors centroids, the clustering algorithm has not worked. Use as a backup KMeans
    if len(list(set(labels))) < cf.number_colors:
        print(method+" has not provided any results. Instead, KMeans method is used instead")
        return cluster(array , "KMeans" )

    # Get all the centroids (note that the clustering is by points and the centroids are computed via the whole array)
    centroids, colours = getCentroids( array , labels )

    # Represent all the pixels via its centroids
    # v.represent_pixels( array , colours )

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

    # Compose the output dictionary
    output = {}
    output["palette"] = palette
    output["palette_labels"] = { l: centroids[l,:] for l in range(centroids.shape[0]) }
    output["labels"] = labels
    output["array"] = array

    return palette


def getCentroids(array, labels):

    # Once the clustering has been carried out, get the centroids based on the points and their labelling
    centroids = np.zeros( (
        max(labels)+1
        , c.dimensions.get(cf.colour_format)
    ) )
    colours = np.zeros( array.shape )

    # Compute for all the elements in the array, the mean coordinates
    for r in range(0,max(centroids.shape)):
        centroids[r,:] = np.mean(
            array[labels == r]
            , axis = 0
        )

    # Assign for each array element the colour of its centroid
    for r in range(0,max(array.shape)):
        colours[r,:] = centroids[labels[r],:]

    return centroids, colours


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


# Remove out-of-range pixels
def removeIncorrectPx( array , format ):

    # Change the size of the array (if necessary)
    if (len(array.shape) == c.dimensions.get(format)):
        array = np.reshape(array,
                           (np.prod(array.shape[0:len(array.shape) - 1]),
                            array.shape[len(array.shape) - 1])
                           )

    # For every pixel, check whether it is valid or not
    if len(array.shape) > 1:
        valid_values = np.array(
            [ f.isInRange( a , c.minimum_values_cluster.get(format) , c.maximum_values_cluster.get(format) ) for a in array ]
        )
    else:
        valid_values = f.isInRange( array , c.minimum_values_cluster.get(format) , c.maximum_values_cluster.get(format) )

    return array[valid_values == True , :]


# Convert to black and white pixels (depending on the colour format)
def extremeColoursCorrection( array , format ):

    # Change the size of the array (if necessary)
    if (len(array.shape) == c.dimensions.get(format)):
        array = np.reshape(array,
                           (np.prod(array.shape[0:len(array.shape) - 1]),
                            array.shape[len(array.shape) - 1])
                           )

    # Black and white conditions
    isBlack = {
        True: False
        , False: lambda x, y: x[2] < y[2]
    }.get(format == "RGB")
    isWhite = {
        True: False
        , False: lambda x, y: ( x[2] > y[2] ) and ( x[1] < y[1] )
    }.get(format == "RGB")

    # For all the elements in the array, convert them to black or white
    colours = np.copy(array)
    colours[
        np.array(
            [ isBlack( a , c.black_colour.get("limits") ) for a in array ]
        ) == True
    ] = c.black_colour.get(format)
    colours[
        np.array(
            [ isWhite( a , c.white_colour.get("limits") ) for a in array ]
        )
    ] = c.white_colour.get(format)

    return colours