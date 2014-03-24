"""
A demo that illustrates how to:
    + Construct the Discrete Karhunen-Loeve expansion of a random field.
    + Sample from it.

Author:
    Ilias Bilionis

Date:
    3/24/2014

"""


import numpy as np
import scipy.weave as weave
import matplotlib.pyplot as plt
import kle


def compute_covariance_matrix(X, s, ell):
    """
    Computes the covariance matrix at ``X``. This simply computes a
    squared exponential covariance and is here for illustration only.
    We will be using covariances from this package:
    `GPy <https://github.com/SheffieldML/GPy>`_.

    :param X:   The evaluation points. It has to be a 2D numpy array of
                dimensions ``num_points x input_dim``.
    :type X:    :class:`numpy.ndarray``
    :param s:   The signal strength of the field. It must be positive.
    :type s:    float
    :param ell: A list of lengthscales. One for each input dimension. The must
                all be positive.
    :type ell:  :class:`numpy.ndarray`
    """
    assert X.ndim == 2
    assert s > 0
    assert ell.ndim == 1
    assert X.shape[1] == ell.shape[0]
    C = np.zeros((X.shape[0], X.shape[0]))
    # We implement the function in C, otherwise it is very slow...
    code = \
"""
double dx;
for(int i=0; i<NX[0]; i++)
for(int j=0; j<NX[0]; j++)
    for(int k=0; k<NX[1]; k++) {
        dx = (X2(i, k) - X2(j, k)) / ELL1(k);
        C2(i, j) += dx * dx;
    }
"""
    weave.inline(code, ['X', 'ell', 'C'])
    return s ** 2 * np.exp(-0.5 * C)


if __name__ == '__main__':
    # Number of input points in x dimension
    n_x = 20
    # Number of inputs points in y dimension
    n_y = 20
    # Size of x dimension
    L_x = 1.
    # Size of y dimension
    L_y = 1.
    # Length scales
    ell = np.array([0.2, 0.2])
    # The signal strength of the field
    s = 1.
    # The percentage of energy of the field you want to keep
    energy = 0.98
    # The points of evaluation of the random field
    x = np.linspace(0, L_x, n_x)
    y = np.linspace(0, L_y, n_y)
    XX, YY = np.meshgrid(x, y)
    X = np.hstack([XX.flatten()[:, None], YY.flatten()[:, None]])
    # Construct the covariance matrix
    C = compute_covariance_matrix(X, s, ell)
    # Compute the eigenvalues and eigenvectors of the field
    eig_values, eig_vectors = np.linalg.eigh(C)
    # Let's sort the eigenvalues and keep only the largest ones
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    # The energy of the field up to a particular eigenvalue:
    energy_up_to = np.cumsum(eig_values) / np.sum(eig_values)
    # The number of eigenvalues giving the desired energy
    i_max = np.arange(energy_up_to.shape[0])[energy_up_to >= energy][0]
    # Plot this
    print 'Ploting energy of the field.'
    plt.figure()
    plt.plot(energy_up_to, 'b', linewidth=2)
    plt.plot(np.ones(energy_up_to.shape[0]), 'r', linewidth=2)
    plt.plot(np.hstack([np.arange(i_max), [i_max] * 50]),
             np.hstack([np.ones(i_max) * energy, np.linspace(0, energy_up_to[i_max], 50)[::-1]]),
             'g', linewidth=2)
    plt.ylim([0, 1.1])
    plt.title('Field Energy', fontsize=16)
    plt.xlabel('Eigenvalue Number', fontsize=16)
    plt.ylabel('Energy', fontsize=16)
    plt.legend(['Truncated expansion energy', 'Full Energy', '98% energy'],
               loc='best')
    print 'Close figure to continue...'
    plt.show()
    # Now let's plot a few eigenvectors
    for i in xrange(3):
        plt.figure()
        c = plt.contourf(XX, YY, eig_vectors[:, i].reshape((n_x, n_y)))
        plt.colorbar(c)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.title('Eigenvector %d' % (i + 1), fontsize=16)
    print 'Close all figures to continue...'
    plt.show()
    # Now, let's construct the D-KLE of the field and sample it.
    d_kle = kle.DiscreteKarhunenLoeveExpansion(X, eig_vectors[:,:i_max+1],
                                               eig_values[:i_max+1])
    print 'Some info about the expansion:'
    print str(d_kle)
    # Let's plot a few samples
    for i in range(5):
        plt.figure()
        c = plt.contourf(XX, YY, d_kle.sample().reshape((n_x, n_y)))
        plt.colorbar(c)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.title('Sample %d' % i, fontsize=16)
    plt.show()
