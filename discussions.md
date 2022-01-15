# Context
The aim of this file is both registering potential conflictive points to discuss, and tracking all the taken decisions during the development of the project.
## Clustering
### Is it possible to determine whether a cluster is well-formed or not?
As KMeans is the algorithm technique which has been chosen to get the clusters (and sub-clusters), the exact amount of them are given as an input of the algorithm. As a result, it may produce clusters whose dispersion is huge. Does this means that these clusters may be subject of a second clustering process (independently from the subclustering process), or it has more to do with the clustering algorithm tuning?
![Wide cluster](/doc/img/cluster.png#gh-dark-mode-only)
### When a clusters collects a reduced amount of pixels... can it be considered as well-formed?
There may be some areas where the density of pixels is reduced. Is it useful to identify this low-density areas, and remove them from clustering? To do so, it may be necessary to define how to implement a density function based on the dimensions used in the clustering (it may be a one-two-three variable function), and trying to, somehow, remove its discrete character. After these regions are excluded, the clustering may be more effective.
![Low density regions](/doc/img/low_density_cluster.png#gh-dark-mode-only)
## Palette translating
### If a cluster is formed by several subclusters, when producing the colour traslation, is it better to work with the centroid of the whole cluster, or working with the centroid of the subcluster?
