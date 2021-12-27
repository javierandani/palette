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
        , [1 , cf.HSV_removal_interval.get("max")/100 , cf.HSV_removal_interval.get("max")/100]
    )
}
minimum_values_cluster = {
    "RGB": [minimum_values.get("RGB")]
    , "HSV": np.add(
        minimum_values.get("HSV")
        , [
            0
            , cf.HSV_removal_interval.get("min") * maximum_values.get("HSV")[1] / 100
            , cf.HSV_removal_interval.get("min") * maximum_values.get("HSV")[2] / 100
        ]
    )
}

# Define the dictionary of number of dimensions
dimensions = {"RGB": 3
              , "HSV": 3
              }

# Define how to represent the pixels (projection, 3d)
projections = {"RGB": "3d"
               , "HSV": "polar"}