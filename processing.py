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
    image_shape = array.shape
    array = np.reshape(array,
                       (np.prod(array.shape[0:len(array.shape) - 1]),
                        array.shape[len(array.shape) - 1])
                       )

    return array, image_shape


def palette_extracting( array , config ):

    # Palette extracting function
    palette_function = {
        "histogram": lambda x,y: histogram(x , y)
        , "clustering": lambda x,y: cluster(x , y )
    }.get(
        config.get("technique")
    )

    # Extract result of the paletteFunction
    output = palette_function( array , config )
    palette_labels = copy.copy(output.get("palette_labels"))   # As it changes with iterations

    # Is needed any sub-clustering?
    for k in palette_labels:

        # Extract cluster's statistics (for every dimension
        stats = {
            cf.colour_format[r]: f.statistics(
                array[ output.get("labels") == k , r ]
            )
            for r in range(0 , output.get("array").shape[1])
        }

        # Determine which dimension/s must be considered to be re-clustered
        dimensions = f.find(
            stats
            , "std_dev"
            , 50
            , lambda x, y: x > y
        )
        n_dimensions = len(dimensions)

        # Re-cluster (if dimensions is greater than 1)
        if n_dimensions > 0:

            # New configuration: change clustering dimensions, add custom options and re-define the number of colours to be extracted
            new_config = copy.copy(config)
            new_config["dimensions"] = dimensions
            new_config["custom"] = {"n_clusters": math.ceil(1.5*n_dimensions) }     # Each dimension can afford 1.5 sub-clusters
            new_config["colours"] = math.ceil(1.5*n_dimensions)                     # Number of colours to obtain

            # Re-cluster
            aux_output = palette_function(
                array[ output.get("labels") == k , : ]
                , new_config
            )

            # Modify current output
            # 1. Add new palette_labels (parent_label + new_label, i.e. "01")
            # 2. Modify labels (parent_label + new_label, i.e. "01")
            output.get("labels")[output.get("labels") == k] = [str(k) + a for a in aux_output.get("labels").astype(str)]
            for p in aux_output.get("palette_labels"):
                output.get("palette_labels")[str(k)+p] = aux_output.get("palette_labels").get(p)

    return output


# Histogram solution (extract N most important colours)
def histogram(array, config):

    # Assign variables
    format = copy.copy(config.get("format"))
    maximum_values = c.maximum_values[format] if len(c.maximum_values[format]) == 3 else c.maximum_values[format][0]*np.ones(3, dtype = np.int8)
    number_bins = cf.number_bins[format] if len(cf.number_bins[format]) == 3 else cf.number_bins[format][0]*np.ones(3, dtype = np.int8)

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
    output = {
        "palette": hist2gram[ range(cf.number_colors) , : ][ : , range(1 , len(a)+1) ]
        , "palette_labels": { h[len(h)-1]: h[range(1,len(h))] for h in hist2gram }
        , "labels": labels
        , "array": array
    }

    return output


def cluster(array, config):

    # Extract variables from the config variable
    format = copy.copy(config.get("format"))           # Colour format
    method = copy.copy(config.get("mode"))             # Clustering technique
    dimensions = copy.copy(config.get("dimensions"))   # Dimensions to cluster by
    number_colours = copy.copy(config.get("colours"))  # Number of maximum clusters
    custom = copy.copy(config.get("custom"))           # Changes respect to the default technique

    # If there is only one dimension, add an auxiliary field
    dimensions.append("Aux") if len(dimensions) == 1 else None

    # Change the size of the array (if necessary)
    if (len(array.shape) == c.dimensions.get(format)):
        array = np.reshape(array,
                           (np.prod(array.shape[0:len(array.shape) - 1]),
                            array.shape[len(array.shape) - 1])
                           )

    # Correct extreme values (white-black) if necessary
    array = extremeColoursCorrection(
        array
        , format
    )

    # Extract points and generate dataFrame
    points = pd.DataFrame(
        columns = list(format)
    )
    for col in points.columns:
        points[col] = array[ : , format.find(col) ]
    points["Aux"] = 0

    # Apply one clustering technique (change any parameter if needed)
    clusteringFunction = c.clustering_function.get( method )
    if bool(custom):  # If not empty
        for att in custom:
            clusteringFunction.__setattr__( att, custom.get(att) )

    # Extract the clustering centers, and the label of each points
    clustering = clusteringFunction.fit( points[dimensions] )
    labels = clustering.labels_

    # If there is less than Number Colors centroids, the clustering algorithm has not worked. Use as a backup KMeans
    if len(list(set(labels))) < number_colours:
        print(method+" has not provided any results. Instead, KMeans method is used instead")
        config["mode"] = "KMeans"
        return cluster(array , config )

    # Get all the centroids (note that the clustering is by points and the centroids are computed via the whole array)
    centroids, colours = getCentroids( array , labels )

    # Represent all the pixels via its centroids
    v.represent_pixels( array , colours )

    # Compose the count of the clusters
    centroid_order = np.zeros((max(centroids.shape), 2))
    for r in range(len(centroids)):
        centroid_order[r, 0] = int(r)
        centroid_order[r, 1] = len(labels[labels == r])

    # Reorder centroids
    centroid_order = centroid_order[np.argsort(centroid_order[:, 1])][::-1]

    # Assign the N first elements to the palette (REVISE??????????)
    palette = []
    [ palette.append(centroids[r, :]) if r in centroid_order[:number_colours, 0] else [] for r in range(0, number_colours) ]
    palette = [ centroids[r in centroid_order[:number_colours,0].astype(np.uint8) ] for r in range(0,number_colours) ]
    palette = np.ndarray( [ number_colours , c.dimensions.get(format) ] )

    # Compose the output dictionary
    output = {
        "palette": palette.astype(np.uint8)
        , "palette_labels": { str(l): centroids[l,:] for l in range(centroids.shape[0]) }
        , "labels": labels.astype(str)
        , "array": array
    }

    return output


def getCentroids(array, labels):

    # Once the clustering has been carried out, get the centroids based on the points and their labelling
    centroids = np.zeros( (
        max(labels)+1
        , c.dimensions.get(cf.colour_format)
    ) )
    colours = np.zeros( array.shape )

    # Compute for all the elements in the array, the mean coordinates
    for r in range(0,centroids.shape[0]):
        centroids[r,:] = np.mean(
            array[labels == r]
            , axis = 0
        )

    # Assign for each array element the colour of its centroid
    for r in range(0,array.shape[0]):
        colours[r,:] = centroids[labels[r],:]

    return centroids, colours


def palette_change(originPalette, config, destinyPalette):

    # Extract variables
    dimensions = config.get("dimensions")
    numberColours = config.get("numberColours")
    exchanges = config.get("exchanges")
    format = config.get("format")
    image_shape = config.get("image_shape")

    # Extract variables from originPalette
    array = originPalette.get("array")
    palette = originPalette.get("palette")
    palette_labels = originPalette.get("palette_labels")
    labels = originPalette.get("labels")

    # Get current dimensions (to be used in translation)
    cols = []
    for d in dimensions:
        cols.append( cf.colour_format.find(d) )

    # Get transform ratios
    ratio = {}
    #palette_names = []
    new_palette = copy.copy(palette)
    for iter in range(numberColours):

        # Compose ratios
        color_new_palette = np.array(list(ImageColor.getcolor( destinyPalette[exchanges[iter][1]] , cf.colour_format )))
        new_palette[iter] = color_new_palette
        ratio[str(exchanges[iter][0])] = list(color_new_palette[cols] - palette[exchanges[iter][0],cols])

        # Compose names of the palettes
        #palette_names = np.append(palette_names, ''.join(str(originPalette[iter,:])))

    # For each element in the image, perform transformation based on ratio, depending on the colour it is assigned to
    image = copy.copy(array)
    image[:,cols] = [
        image[r,cols] + ratio.get(labels[r][0]) if ratio.get(labels[r][0]) != None else image[r,cols] for r in range(0,image.shape[0])  # labels[r][0] to get the clusters ID (first character)
    ]

    # Establish boundaries at the image
    image = normalizeImageFunction( image , cf.colour_format )

    # Cast the whole image to int, and resize
    image = image.astype(np.uint8)
    image = np.reshape(image, image_shape)

    return image, new_palette


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