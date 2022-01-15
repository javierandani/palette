import os
import config as cf
import constants as c
import processing as p
import visualization as v

# Load all images
filenames = os.listdir( os.getcwd() + "\img\." )

for files in filenames:

    # Extract image (RGB)
    image = p.read_image(
        os.getcwd() + "\img\\" + files
        , cf.colour_format
    )

    # Extract array of pixels
    array, image_shape = p.extract_pixels(
        image
        , cf.sampling_rate
    )

    # Represent pixels
    v.represent_pixels( p.removeIncorrectPx( array , cf.colour_format ) )

    # Palette extracting
    paletteInstance = p.palette_extracting(
        array
        , {
            "technique": cf.palette_extracting_mode                     # Histogram / ML
            , "mode": cf.clustering_mode                                # Clustering technique
            , "format": cf.colour_format                                # Colour format
            , "dimensions": c.dataFrameColumns.get(cf.colour_format)    # Dimensions to cluster by
            , "colours": cf.number_colors                               # Number of colours
            , "custom": {}                                              # Non-default options
        }
    )

    # Plot both image and colour palette
    v.figure_palette_plot(files.split(".")[0] + "_" + cf.palette_extracting_mode + "_palette", image, paletteInstance.get("palette"))

    # Transform changing palette
    new_image, new_palette = p.palette_change(
        paletteInstance
        , {
            "dimensions": c.dataFrameColumns.get(cf.colour_format)      # Dimensions to translate by
            , "numberColours": 1                                        # Colours to translate (up to cf.number_colours)
            , "exchanges": c.colour_exhanges                            # Colours to translate
            , "format": cf.colour_format                                # Format of the origin palette
            , "image_shape": image_shape                                # Shape of the image (after re-sampling)
        }
        , cf.default_palette
    )

    # Represent new image
    v.figure_palette_plot( files.split(".")[0] + "new_palette" , new_image , new_palette)