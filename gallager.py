#!/usr/bin/env python
# coding: utf-8


import numpy as np
def parity_check_matrix(n, d_v, d_c, seed=None):
    """
    Builds a regular Parity-Check Matrix H (n, d_v, d_c) following
    Callager's algorithm.

    Parameters:

     n: Number of columns (Same as number of coding bits)
     d_v: number of ones per column (number of parity-check equations including
     a certain variable)
     d_c: number of ones per row (number of variables participating in a
     certain parity-check equation);

    Errors:

     The number of ones in the matrix is the same no matter how we calculate
     it (rows or columns), therefore, if m is
     the number of rows in the matrix:

     m*d_c = n*d_v with m < n (because H is a decoding matrix) => Parameters
     must verify:


     0 - all integer parameters
     1 - d_v < d_v
     2 - d_c divides n

    ---------------------------------------------------------------------------------------

     Returns: 2D-array (shape = (m, n))

    """
    rnd = np.random.RandomState(seed)
    if n % d_c:
        raise ValueError("""d_c must divide n. help(coding_matrix)
                            for more info.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.
                            help(coding_matrix) for
                            more info.""")

    m = (n * d_v) // d_c

    Set = np.zeros((m//d_v, n), dtype=int)
    a = m // d_v

    # Filling the first set with consecutive ones in each row of the set

    for i in range(a):
        for j in range(i * d_c, ((i+1) * d_c)):
            Set[i, j] = 1

    # Create list of Sets and append the first reference set
    Sets = []
    Sets.append(Set.tolist())

    # reate remaining sets by permutations of the first set's columns:
    i = 1
    for i in range(1, d_v):
        newSet = rnd.permutation(np.transpose(Set)).T.tolist()
        Sets.append(newSet)

    # Returns concatenated list of sest:
    H = np.concatenate(Sets)
    return H

