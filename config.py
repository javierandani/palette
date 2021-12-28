# Sampling rate (histogram)
sampling_rate = 5

# Sampling rate (image representation)
_sampling_rate = 2

# Number of clusters
number_bins = {"RGB": [3]
              , "HSV": [16,3,3]
              }

# Number of colours of the palette
number_colors = 6

# Default colour palette
default_palette = ["#DAF7A6","#FFC300","#FF5733","#C70039","#900C3F","#581845"]

# Colour format to produce both clustering and transformations -> ["RGB","HSV"]
colour_format = "HSV"

# Saturation and value extremse removal interval (HSV pre-processing)
removal_interval = {"min": 10, "max": 100}

# When changing the colour palette, define the intensity of the change ["soft","medium","hard"]
change_mode = "soft"

# Clustering technique
clustering_technique = "DBSCAN"

# Maximum iteration (clustering algorithms)
maximum_iterations = 500