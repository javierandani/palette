import os
import config as cf
import processing as p
import visualization as v

# Load all images
filenames = os.listdir(os.getcwd() + "\img\.")

for files in filenames:

    # Extract image (RGB)
    image = p.read_image( os.getcwd() + "\img\\" + files , cf.colour_format )

    # Extract array of pixels
    array = p.extract_pixels(image, cf.sampling_rate)

    # Represent pixels
    v.represent_pixels(array)

    # Color histogram
    palette = p.histogram(array)

    # ML Clustering
    palette_clustering = p.cluster( array , cf.clustering_technique )

    # Plot both image and colour palette
    v.figure_palette_plot(files.split(".")[0] + "_histogram_palette", image, palette)
    v.figure_palette_plot(files.split(".")[0] + "_cluster_palette", image, palette_clustering)

    # Transform changing palette
    new_image, new_palette = p.palette_change( image, palette, cf.default_palette)

    # Represent new image
    v.figure_palette_plot( files.split(".")[0] + "new_palette" , new_image , new_palette)