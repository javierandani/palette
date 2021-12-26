import numpy as np

# Nearest point to (point) between all references
def nearest_point(point, references):

    # Declare distances vector
    distances = []

    # For each element in references
    for d in range(references.shape[0]-1):

        # Get colour
        color = references[d,:]

        # Compute distances (square root of the sum of the differences between coordinates)
        distances = np.append(distances
                              , np.sqrt(
                                  np.sum((point - color) ** 2
                                         , axis = 0
                                    )
                                )
                              )

    # Return de associated point. The nearest is selected
    return references[np.where(distances == np.min(distances))[0][0],:]
