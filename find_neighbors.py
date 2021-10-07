def find_neighbors(pindex, triang):
    """
    Find neighbor vertices of a given vertex in a voronoi diagram.
    
    Parameters
    ----------
    pindex = index of point in delaunay triangulation
    triang = delaunay triangulation of a data set
    
    Returns
    -------
    Indices of the points in triang which are directly adjacent to point at pindex
    """
    neighbors = list()
    for simplex in triang.vertices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
    return list(set(neighbors))
