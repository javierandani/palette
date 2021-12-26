import numpy as np
import cv2
from PIL import ImageColor
from skimage.transform import downscale_local_mean
import constants as c
import config as cf
import functions as f
import conversions as convers

# Read image
def read_image(files , format):

    return {
        "RGB": cv2.imread( files )[:, :, ::-1].astype(np.uint8)
        , "HSV": cv2.cvtColor( cv2.imread( files ) , cv2.COLOR_RGB2HSV )
    }.get(format)


# Extract all points from image into a nx3 vector
def extract_pixels(image, factor):

    # Change image format to RGB
    image = convers.change_image_format(image, cf.colour_format, "RGB")

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