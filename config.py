global sampling_rate, _sampling_rate, number_bins, number_colors, default_palette, colour_format

# Sampling rate (histogram)
sampling_rate = 5

# Sampling rate (image representation)
_sampling_rate = 2

# Number of clusters
number_bins = {"RGB": [16]
              , "HSV": [16,3,3]
              }

# Number of colours of the palette
number_colors = 5

# Default colour palette
default_palette = ["#DAF7A6","#FFC300","#FF5733","#C70039","#900C3F","#581845"]

# Colour format to produce both clustering and transformations -> ["RGB","HSV"]
colour_format = "RGB"