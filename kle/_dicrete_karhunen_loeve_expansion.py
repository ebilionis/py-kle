"""
A class representing a dicrete Karhunen Loeve Expansion.

Author:
    Ilias Bilionis

Date:
    3/23/2014

"""


__all__ = ['DiscreteKarhunenLoeveExpansion']


import numpy as np
import scipy.stats


class DiscreteKarhunenLoeveExpansion(object):

    """
    A class representing a discrete Karhunen-Loeve Expansion.

    Specific Karhunen-Loeve expansions inherit from this class.

    :param points:      The points on which the D-KLE can be evaluated.
                        It is assumed that the dimension of this is
                        ``num_points x input_dim``.
    :type points:       :class:`numpy.ndarray`
    :param eig_vectors: The matrix of the eigenvectors of D-KLE. This is
                        assumed to be of dimensions ``num_points x num_terms``.
    :type eig_vectors:  :class:`numpy.ndarray`
    :param eig_values:  The eigenvalues of the D-KLE. The size of this should
                        be ``num_terms`` and of course all the eigenvalues
                        should be positive.
    :type eig_values:   :class:`numpy.ndarray`
    :param name:        A name for the object.
    :type name:         str
    """

    # The points on which D-KLE is evaluated
    _points = None

    # The eigenvectors
    _eig_vectors = None

    # The eigenvalues
    _eig_values = None

    @property
    def points(self):
        """
        Get the points on which KLE can be evaluated.
        """
        return self._points

    @property
    def eig_vectors(self):
        """
        Get the eigenvectors.
        """
        return self._eig_vectors

    @property
    def eig_values(self):
        """
        Set the eigenvalues.
        """
        return self._eig_values

    @property
    def num_points(self):
        """
        Get the number of points on which we do the evaluation.
        """
        return self.points.shape[0]

    @property
    def input_dim(self):
        """
        Get the number of input dimensions.
        """
        return self.points.shape[1]

    @property
    def num_terms(self):
        """
        Get the number of terms of the D-KLE
        """
        return self.eig_vectors.shape[1]

    def __init__(self, points, eig_vectors, eig_values, name='D-KLE'):
        """
        Initialize the object.
        """
        points = points[:, None] if points.ndim == 1 else points
        assert points.ndim == 2
        eig_vectors = eig_vectors[:, None] if eig_vectors.ndim == 1 else eig_vectors
        assert eig_vectors.ndim == 2
        eig_values = eig_values.flatten() if eig_values.ndim == 2 else eig_values
        assert eig_values.ndim == 1
        assert points.shape[0] == eig_vectors.shape[0]
        assert eig_vectors.shape[1] == eig_values.shape[0]
        assert np.all(eig_values > 0)
        self._points = points
        self._eig_vectors = eig_vectors
        self._eig_values = eig_values
        self.__name__ = name

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = self.__name__ + '\n'
        s += 'Input dim: ' + str(self.input_dim) + '\n'
        s += 'Num. Points: ' + str(self.num_points) + '\n'
        s += 'Num. Terms: ' + str(self.num_terms)
        return s

    def __call__(self, xi):
        """
        Evaluates the D-KLE at ``xi``.

        :param xi:  The D-KLE coefficients.
        :type xi:   :class:`numpy.ndarray`
        """
        assert xi.ndim <= 2
        if xi.ndim == 1:
            assert xi.shape[0] == self.num_terms
        elif xi.ndim == 2:
            assert xi.shape[1] == self.num_terms
        return np.dot(xi * np.sqrt(self.eig_values), self.eig_vectors.T)

    def sample(self, size=1):
        """
        Sample random fields.
        """
        xi = np.random.normal(0, 1, size=(size, self.num_terms))
        return self(xi)

    def project(self, X):
        """
        Project the vector y to the D-KLE basis.

        :param X:   A vector in the original space.
        :type X:    :class:`numpy.ndarray`
        """
        assert X.ndim <= 2
        if X.ndim == 1:
            assert X.shape[0] == self.num_points
            return np.dot(self.eig_vectors.T, X / np.sqrt(self.eig_values))
        elif X.ndim == 2:
            assert X.shape[1] == self.num_points
            return np.array([self.project(X[i, :]) for x in xrange(X.shape[0])])
