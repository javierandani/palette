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

    # Plot image (transform to RGB space)
    axFigure.imshow(
        convers.image_conversion(image, cf.colour_format, "RGB")
    )

    # Plot palette colour
    for iter in range(len(palette)):
        x = [iter, iter + 1]
        y = [1, 1]
        axPalette.fill_between(x, y, color = palette[iter])     # Area chart
        #plt.show(block = False)                                 # Refresh image

    # Remove axis
    axFigure.axis('off')
    axPalette.axis('off')

    # Save image
    if saveFig:
        plt.savefig('output/'+fileName)


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


def represent_pixels(array, *args):

    # Normalize colours (if there is more than 1 argument, then use the first one as input)
    if len(args) == 0:
        colours = convers.normalize(
            np.array( [ convers.colour_conversion( a , cf.colour_format , "RGB") for a in array ] )
            , "RGB"
            , True
        )
    else:
        colours = convers.normalize(
            np.array( [ convers.colour_conversion( a , cf.colour_format , "RGB") for a in args[0] ] )
            , "RGB"
            , True
        )

    # Maximum values
    maximum_values = c.maximum_values.get( cf.colour_format )

    # Represent figure
    fig = plt.figure()
    ax = fig.add_subplot(projection = c.projections[cf.colour_format])

    # Config and scatter functions
    config = {
        "polar":
            {
                "theta": array[:,0] * 2 * np.pi / maximum_values[0]
                , "radius": array[:,1] / np.max(array[:,1])
                , "colours": colours # convers.normalize( array , cf.colour_format , True ) # array[:,0] * 2 * np.pi / np.max(array[:,0])
                , "area": 200 * (array[:,1] / np.max(array[:,1])) ** 2
                , "cmap": "RGB"
                , "alpha": None
            }
        , "3d":
            {
                "x": array[:,0]
                , "y": array[:,1]
                , "z": array[:,2]
                , "colours": convers.normalize( colours , "RGB" , True )
                , "cmap": cf.colour_format
            }
    }.get(c.projections[cf.colour_format])
    represent_function = {
        "polar": lambda x, y: polar_graph( x , y )
        , "3d": lambda x, y: scatter_graph( x , y )
    }.get(c.projections[cf.colour_format])

    # Execute functions
    represent_function(
        ax
        , config
    )
    plt.show(block = False)


def polar_graph( ax , config ):

    ax.scatter(
        config["theta"]
        , config["radius"]
        , c = config["colours"]
        , s = config["area"]
        , cmap = config["cmap"]
        , alpha = config["alpha"]
    )


def scatter_graph( ax , config ):

    ax.scatter(
        config["x"]
        , config["y"]
        , config["z"]
        , c = config["colours"]
        , cmap = config["cmap"]
    )