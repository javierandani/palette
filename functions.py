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


def isInRange(value, min_value, max_value):


    # Check whether the input are arrays
    if type(value) == np.ndarray:

        # If there is more than 1 point, then check all of them
        if len(value.shape) > 1:

            # Initialize the vector
            valid = np.zeros((value.shape[0],1))

            # Convert the max_value and min_value's shape to the one of each element of value
            if len(max_value) != len(value[0,:]):
                max_value = [max_value[0]] * len(value)
            if len(min_value) != len(value[0,:]):
                min_value = [min_value[0]] * len(value)

            # For every pixel, check that all the components are in range
            for i in range(0,value.shape[0]):
                valid[i] = all( value[i,v] == np.clip( value[i,v] , min_value[v] , max_value[v] ) for v in range(0,value.shape[1]) )

        else:

            # Convert the max_value and min_value's shape to the one of each element of value
            if len(max_value) != len(value):
                max_value = [max_value[0]] * len(value)
            if len(min_value) != len(value):
                min_value = [min_value[0]] * len(value)

            # Initialize the vector
            valid = all( value[v] == np.clip( value[v] , min_value[v] , max_value[v] ) for v in range(0,value.shape[0]) )

    else:

        valid = value == np.clip( value, min_value , max_value)

    return valid


def statistics(x):

    # Define main statistics of a variable
    output = {
        "mean": np.mean(x)
        , "std_dev": np.std(x)
        , "var": np.var(x)
        , "median": np.median(x)
    }

    return output


def find(lst, key, value, condition):

    # Define output list
    output = []
    for i, dic in enumerate(lst):
        if condition(lst[dic][key],value):
            output.append(dic)

    return output