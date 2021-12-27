import numpy as np
import constants as c
import colorsys
from colormap import rgb2hex
import cv2

# Colour conversion
def colour_conversion(colour, originFormat, destinyFormat):

    # Create a tuple-indexed dictionary
    conversionFunction = {
        ("RGB","RGB"): lambda x: x
        , ("RGB","HSV"): lambda x: normalize( rgb2hsv(x) , "HSV" , False)
        , ("RGB","HEX"): lambda x: rgb2hex( x[0].astype(np.uint8) ,x[1].astype(np.uint8),x[2].astype(np.uint8))
        , ("HSV","RGB"): lambda x: normalize( hsv2rgb(x) , "RGB" , False)
        , ("HSV","HSV"): lambda x: x
        , ("HSV","HEX"): lambda x: _rgb2hex( normalize( hsv2rgb(x) , "RGB" , False ).astype(int) )
        , ("HEX","RGB"): hex2rgb
        , ("HEX","HSV"): hex2hsv
        , ("HEX","HEX"): lambda x: x
    }.get((originFormat, destinyFormat))

    # If it is an array, then iterate
    if len(colour.shape) > 1:
        return [conversionFunction(c) for c in colour]
    else:
        return conversionFunction(colour)


# Colour conversion
def image_conversion(image, originFormat, destinyFormat):

    # Create a tuple-indexed dictionary
    conversionFunction = {
        ("RGB", "RGB"): lambda x: x
        , ("RGB", "HSV"): lambda x: cv2.cvtColor( x , cv2.COLOR_RGB2HSV )
        , ("HSV", "RGB"): lambda x: cv2.cvtColor( x , cv2.COLOR_HSV2RGB )
        , ("HSV", "HSV"): lambda x: x
    }.get((originFormat, destinyFormat))

    return conversionFunction(image)


# HEX to RGB
def hex2rgb(colour):

    colour = colour.lstrip('#')
    lv = len(colour)
    return tuple(int(colour[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


# HEX to HSV
def hex2hsv(colour):

    # If it a string, there is only a colour
    if type(colour) == str:
        rgb = hex2rgb(colour)
        hsv = normalize( rgb2hsv(rgb) , "HSV" , False )
    else:
        rgb = [ hex2rgb(c) for c in colour ]
        hsv = [ normalize( rgb2hsv(r) , "HSV" , False ) for r in rgb]

    return hsv


# HSV to RGB (normalized)
def hsv2rgb(hsv):

    # Normalize (if not yet)
    hsv = normalize(hsv, "HSV", True) if(max(hsv) > 1) else hsv

    return np.array( colorsys.hsv_to_rgb( hsv[0] , hsv[1] , hsv[2] ) )


# RGB to HSV (normalized)
def rgb2hsv(rgb):

    # Normalize (if not yet)
    if(max(rgb) > 1):
        rgb = np.divide(
            rgb
            , c.maximum_values["RGB"]
        )

    return colorsys.hsv_to_rgb( rgb[0] , rgb[1] , rgb[2] )


def _rgb2hex(rgb):

    return rgb2hex( rgb[0] , rgb[1] , rgb[2] )


# Change image coding
def change_image_format(image, originFormat, destinyFormat):

    # Create a tuple-indexed dictionary
    conversionFunction = {
        ("RGB","RGB"): lambda x: x
        , ("RGB","HSV"): lambda x: cv2.cvtColor(x , cv2.COLOR_RGB2HSV )
        , ("HSV","RGB"): lambda x: cv2.cvtColor(x , cv2.COLOR_HSV2BGR )
        , ("HSV","HSV"): lambda x: x
    }.get((originFormat, destinyFormat))

    return conversionFunction(image)


# Normalize (or de-normalize colours)
def normalize(colour, format, normalize):

    # Norm function
    normFunction = {
        True: lambda x: np.divide(x, c.maximum_values.get(format) )
        , False: lambda x: np.multiply(x, c.maximum_values.get(format) )
    }.get(normalize)

    # If it is an array, then iterate
    if len(colour.shape) > 1:
        return np.array([normFunction(c) for c in colour])
    else:
        return normFunction(colour)