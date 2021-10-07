def voronoi_volumes(points):
    """ 
    Input = data points to be used in Voronoi diagram. Numpy array
    Returns = Volume (areas in 2D) of cells as well as the corresponding indices of the 
    points inside the cell
    """
    v = spatial.Voronoi(points)
    vol = np.zeros(v.npoints)
    index = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        index[i] = reg_num
        if -1 in indices:
            vol[i] = np.inf
        else:
            vol[i] = spatial.ConvexHull(v.vertices[indices]).volume
    return vol/(scale**2),index
