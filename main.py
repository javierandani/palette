import os
import config as cf
import processing as p
import visualization as v

# Load all images
filenames = os.listdir( os.getcwd() + "\img\." )

for files in filenames:

    # Extract image (RGB)
    image = p.read_image( os.getcwd() + "\img\\" + files , cf.colour_format )

    # Extract array of pixels
    array = p.extract_pixels(image, cf.sampling_rate)

    # Represent pixels
    v.represent_pixels(p.removeIncorrectPx( array , cf.colour_format ))

    # Palette extracting
    paletteInstance = p.palette_extracting(
        array
        , {
            "technique": cf.palette_extracting_mode
            , "mode": cf.clustering_mode
        }
    )

    # Plot both image and colour palette
    v.figure_palette_plot(files.split(".")[0] + "_" + cf.palette_extracting_mode + "_palette", image, paletteInstance.get("palette"))

    # Transform changing palette
    #new_image, new_palette = p.palette_change( image, palette, cf.default_palette)

    # Represent new image
    #v.figure_palette_plot( files.split(".")[0] + "new_palette" , new_image , new_palette)