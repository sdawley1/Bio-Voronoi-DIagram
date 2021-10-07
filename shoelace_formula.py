def shoelace(x_y):
    """
    Input = Numpy array
    Returns = area of cell using shoelace formula
    """
    x_y = np.array(x_y)
    x_y = x_y.reshape(-1,2)
    x = x_y[:,0]
    y = x_y[:,1]
    S1 = np.sum(x*np.roll(y, -1))
    S2 = np.sum(y*np.roll(x, -1))
    area = .5*np.absolute(S1 - S2)
    return area/(scale**2)
