import constants as c
import config as cf
import conversions as convers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Plot image and save it
def image_palette_plot(fileName, image, palette, saveFig):

    # Compose figure
    fig = plt.figure()

    # Determine height ratios through gridspecs (75%-25%)
    gs = gridspec.GridSpec(2, 1, height_ratios = [3,1])
    axFigure = plt.subplot(gs[0])
    axPalette = plt.subplot(gs[1])

    # Suplitle (without file extension)
    fig.suptitle(fileName.split('.')[0])

    # Plot image (transform to RGB space). Only use cmap if the colour format is not RGB
    {
        True: axFigure.imshow( image )
        , False: None if cf.colour_format == "RGB" else axFigure.imshow( image , cmap = str.lower(cf.colour_format) )
    }.get(cf.colour_format == "RGB")

    # Plot palette colour
    for iter in range(len(palette)):
        x = [iter, iter + 1]
        y = [1, 1]
        axPalette.fill_between(x, y, color = palette[iter])     # Area chart
        plt.show(block = False)                                 # Refresh image

    # Remove axis
    axFigure.axis('off')
    axPalette.axis('off')

    # Save image
    if saveFig:
        plt.savefig('img/'+fileName)


# Represent the colour palette for this image
def figure_palette_plot(fileName, image, palette):

    # If the palette has more than N dimensions, get the last ones
    if (palette.shape[1] > 3):
        palette = palette[: , palette.shape[1]-c.dimensions : palette.shape[1]]

    # Change colour palette
    colours = [convers.colour_conversion( p , cf.colour_format , "HEX" ) for p in palette]

    # Plot both figure and palette
    image_palette_plot(fileName,    # File name
                       image,       # Image (matrix)
                       colours,     # Colours palette
                       True)        # Save Figure


def represent_pixels(array):

    # Normalize colours
    colours = np.array( [ convers.colour_conversion( a , cf.colour_format , "RGB") for a in array ] )

    # Represent figure
    fig = plt.figure()
    ax = fig.add_subplot(projection = c.projections[cf.colour_format])
    {
        "polar": ax.scatter(
                array[:,0]
                , array[:,1] / np.max(array[:,1])
                , c = convers.normalize( array , "RGB" , True )
                , cmap = "RGB"
            )
        , "3d": ax.scatter(
                array[:,0]
                , array[:,1]
                , array[:,2]
                , c = convers.normalize( colours , "RGB" , True )
                , cmap = "RGB"
            )
    }.get(c.projections[cf.colour_format])
    plt.show(block = False)

