

def reshape_to_4d(tensor):
    if len(tensor.shape) == 3:
        """ 
        if data is two dimensional: wrap each data case inside an extra single dimension (channel)
        """
        return tensor.reshape((tensor.shape[0], 1) + (tensor.shape[1:]))

    # else len == 2 always
    """ 
    if data is one dimensional: wrap channel as above AND
    transform 1d arrays of length n onto arrays of shape (1, n)
    """
    return tensor.reshape((tensor.shape[0], 1, 1, tensor.shape[1]))
