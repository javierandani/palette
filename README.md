# APAIDEX
**A**utomatic **PA**lette **ID**entification and **EX**change is an amateur Python-based project, whose aim is developing several image processing techniques in order to identify the colour palette of a given image, from basic histogram definition to ML clustering techniques.
Note that this project is Windows OS native.

## Initial considerations
As the image codification may vary, the code is prepared to read and process images in both RGB and HSV formats. In consequence, all the functions are customized and, depending on the format, their behaviour may vary.
The code is structured in several files:
1. __config__, where all the different customizing options, from the colour format to process the data to the clustering technique (if any).
2. __constants__ defines all the values which may be needed in the image processing process. Several of them are dependant to some options defined on the __config__ file.
3. __main__, where the core of the program is  described. This file acts like an anchor to the rest, as it calls processes from all the different files.
4. __processing__ collects all the functions which support the program's execution, from the image reading to colour translation, through pixels' clustering. Here is where the strong agnostic spirit of the project (regarding colour format) is present the most.
5. __functions__ are auxiliar functions which may be needed at the processing.
6. __visualization__ helps the user to represent graphically the information which is being processed, from the bare images to their colour palette, and the pixels' distribution.
7. __conversions__ provide the needed flexibility in order to work with different colour formats, as it allows to transform both pixels-based arrays to proper images.

## Clustering approach
As it has been mentioned, the main objective of this project is extracting the colour palette of a certain image. To perform this, a ML clustering approach has been chosen.
It is true that, via the __config__ file, it is possible to choose from a ML approach, or a simpler one (_palette_extracting_mode_: histogram). Nevertheless, the last one corresponds to an initial version of the code, despite it is possible to perform clustering via this option.
### Histogram Clustering
This histogram clustering is based on defining equal-size bins into the colour space. After this, each pixel is associated to a bin (defined from its center), and the most common ones define the colour palette. Depending on the _number_bins_ parameter, it may produce too homogeneous results (if this parameter is greater regarding the colour space), or too heterogeneous (if it is small, it may produce very different central colours).
This approach does not seem to be the most appropiate, as the pixels of an image does not distribute uniformly. Thus, a ML Clustering way arised as an interesting alternative, in order to provide a more accurate result.
### ML Clustering
As Python provides several options, when dealing with clustering, there is a collection of the most popular algorithms in the __constants__ file. The selection of one algorithm or another depends on the _clustering_technique_ variable (tuneable at __config__).
After a recursive try and watch process, the _KMeans_ algorithm has been stated as the preferred option. Nevertheless, there is a huge variety of algorithms to choose, and customize.
Indeed, not only a one-layer clustering is perfomed, but two. This means that, when clustering the pixels of an image (let's say, in HSV format), a second-level clustering is performed, if the distribution of values leads to do it.

## Palette change
Another objective of this project, which can not be achieved without the first one, is changing certain colours (of tones) of the image, by others, providing a new image. This part is being discussed, as there may appear some doubts about the best way of implementing it.
