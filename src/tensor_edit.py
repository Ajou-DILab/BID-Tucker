"""
Title: 
Authors:
- Lee Sael (sael@ajou.ac.kr) Ajou University
- Sang Suk Lee, Ajou University
- HeaWon Moon, Ajou University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

import tensorly as tl
import numpy as np
from tensorly.base import vec_to_tensor

'''originally copied form Tensorly and modified'''

def mode_dot_F(tensor, matrix_or_vector, mode, transpose=False):
    """
    [Updated function orginially from tensorly]
    n-mode product of a tensor and a matrix or vector at the specified mode

    Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`

    Parameters
    ----------
    tensor : ndarray
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrix_or_vector : ndarray
        1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
        matrix or vectors to which to n-mode multiply the tensor
    mode : int
    transpose : bool, default is False
        If True, the matrix is transposed.
        For complex tensors, the conjugate transpose is used.

    Returns
    -------
    ndarray
        `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

    See also
    --------
    multi_mode_dot : chaining several mode_dot in one call
    """
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    if tl.ndim(matrix_or_vector) == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned in mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[dim]} (dim 1 of matrix)"
            )

        if transpose:
            matrix_or_vector = tl.conj(tl.transpose(matrix_or_vector))

        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif tl.ndim(matrix_or_vector) == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned for mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[0]} (vector size)"
            )
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            # Ideally this should be (), i.e. order-0 tensors
            # MXNet currently doesn't support this though..
            new_shape = []
        vec = True

    else:
        raise ValueError(
            "Can only take n_mode_product with a vector or a matrix."
            f"Provided array of dimension {T.ndim(matrix_or_vector)} not in [1, 2]."
        )

    res = tl.dot(matrix_or_vector, unfold_F(tensor, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return vec_to_tensor(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return fold_F(res, fold_mode, new_shape)

def multi_mode_dot_F(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
    """
    [Updated function orginially from tensorly]
    n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of length ``tensor.ndim``

    skip : None or int, optional, default is None
        If not None, index of a matrix to skip.
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        If True, the matrices or vectors in in the list are transposed.
        For complex tensors, the conjugate transpose is used.

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list at mode `mode`

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:

    :math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`

    See also
    --------
    mode_dot
    """
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue

        if transpose:
            res = mode_dot_F(res, tl.conj(tl.transpose(matrix_or_vec)), mode - decrement)
        else:
            res = mode_dot_F(res, matrix_or_vec, mode - decrement)

        if tl.ndim(matrix_or_vec) == 1:
            decrement += 1

    return res


def unfold_F(tensor, mode):
    """
    [Updated function orginially from tensorly]
    Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return tl.reshape(tl.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1),'F')



def fold_F(unfolded_tensor, mode, shape):
    """
    [Updated function orginially from tensorly]
    Refolds the mode-`mode` unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return tl.moveaxis(tl.reshape(unfolded_tensor, full_shape,'F'), 0, mode)


def print_recon_error(core, factor, _orgten, ord_lst=[], DEBUG=False): 
    for ord in ord_lst:
        core  = mode_dot_F(core, factor[ord].T, ord)
    
    new_tensor = multi_mode_dot_F(core, factor)
    error = 100000
    if(len(_orgten)>0 and _orgten.shape == new_tensor.shape):
        error = np.linalg.norm(_orgten - new_tensor)
        nerror = error ** 2 / np.linalg.norm(_orgten) ** 2
        if(DEBUG): 
            print(f'check close: {np.allclose(_orgten,new_tensor)} with normalized error: {nerror}')

    return error, new_tensor


def create_tensor_stream(X, start_to_stream, batch_sizes=[], svd_rank=20):
        # generate np array of batch sizes
        if start_to_stream < 0:
                batch_size = int(-start_to_stream)
                start_to_stream = X.shape[0] % batch_size 
                if start_to_stream == 0:        # no remainder 
                        start_to_stream = batch_size
                        batch_sizes = np.full(((X.shape[0]-batch_size) // batch_size), batch_size, dtype=int)
                else: # if there is a remainder
                        if(start_to_stream<svd_rank): 
                                start_to_stream = start_to_stream + batch_size
                                batch_sizes = np.full((X.shape[0] // batch_size) - 1 , batch_size, dtype=int)
                        else:
                                batch_sizes = np.full((X.shape[0] // batch_size ), batch_size, dtype=int)        

        total_batch_size = np.sum(batch_sizes)
        if X.shape[0] != start_to_stream + total_batch_size:
                raise ValueError('Total batch size should be the size of streaming part of the tensor.')

        X_stream = [X[:start_to_stream]]
        batch_start = start_to_stream
        for batch_size in batch_sizes:
                batch_end = batch_start + batch_size
                X_stream.append(X[batch_start:batch_end])
                batch_start = batch_end
        
        return X_stream

