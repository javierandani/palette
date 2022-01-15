# Sampling rate
sampling_rate = 1

# Sampling rate (image representation)
_sampling_rate = 2

# Number of clusters
number_bins = {"RGB": [3]
              , "HSV": [16,3,3]
              }

# Number of colours of the palette
number_colors = 5

# Default colour palette
default_palette = ["#DAF7A6","#FFC300","#FF5733","#C70039","#900C3F","#581845"]

# Colour format to produce both clustering and transformations -> ["RGB","HSV"]
colour_format = "HSV"

# Saturation and value extremse removal interval (HSV pre-processing)
removal_interval = {"min": 10, "max": 100}

# When changing the colour palette, define the intensity of the change ["soft","medium","hard"]
change_mode = "soft"

# Palette extracting mode ["histogram","clustering"]
palette_extracting_mode = "clustering"

# Clustering technique
clustering_mode = "KMeans"

# Minimum percentage of pixels to conform a cluster
pct_min_px_cluster = 5

# Maximum iteration (clustering algorithms)
maximum_iterations = 500