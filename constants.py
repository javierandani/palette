global sampling_rate, _sampling_rate, number_bins, rgb_max_value, hsv_max_value, number_colors, default_palette, palette, colour_format, _format

# Sampling rate (histogram)
#sampling_rate = 5

# Sampling rate (image representation)
#_sampling_rate = 2

# Number of clusters
#number_bins = {"RGB": [16]
#              , "HSV": [16,3,3]
#              }

# Maximum values
maximum_values = {"RGB": [255]
                  , "HSV": [360, 255, 255]
                  }

# Minimum values
minimum_values = {"RGB": [0]
                  , "HSV": [0, 0, 0]
                  }

# Number of colours of the palette
#number_colors = 5

# Default colour palette
#default_palette = ["#DAF7A6","#FFC300","#FF5733","#C70039","#900C3F","#581845"]


# Colour format to produce both clustering and transformations -> ["RGB","HSV"]
#colour_format = "RGB"


# Define the dictionary of number of dimensions
dimensions = {"RGB": 3
              , "HSV": 3
              }

# Define how to represent the pixels (projection, 3d)
projections = {"RGB": "3d"
               , "HSV": "polar"}