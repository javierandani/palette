import config as cf
import numpy as np

# Maximum values
maximum_values = {
    "RGB": [255]
    , "HSV": [360 , 255 , 255 ]
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