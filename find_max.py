def find_max(arr):
    """
    Input = 1D Numpy array
    Returns = Maximum value in the array
    """
    maximum = 0
    for n in arr:
        if n != np.inf and n > maximum:
            maximum = n
    return maximum
