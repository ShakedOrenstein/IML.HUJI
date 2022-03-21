from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        if len(X) <= 1:
            self.var_ = 0

        if self.biased_:
            self.var_ = np.var(X, ddof=0)
        else:
            self.var_ = np.var(X, ddof=1)

        self.fitted_ = True
        return self

    @staticmethod
    def _uni_normal_pdf(x: float, mu: float, var: float):
        """
        Computes the pdf of a Uni Gaussian normal sample x.
        parameters
        ----------
        x: float, the sample
        mu: the expected value of x
        var: the variance of x
        """

        numerator: float = np.e ** ((-(x - mu) ** 2) / (2 * var))
        return numerator / np.sqrt(var * 2 * np.pi)

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        normal_pdf_vector: np.Array = np.vectorize(UnivariateGaussian._uni_normal_pdf)
        return normal_pdf_vector(X, self.mu_, self.var_)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        normal_pdf_vector: np.Array = np.vectorize(UnivariateGaussian._uni_normal_pdf)
        pdf_results_vector: np.Array = normal_pdf_vector(X, mu, sigma)
        return np.sum(np.log(pdf_results_vector))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)

        self.fitted_ = True
        return self

    @staticmethod
    def _multi_normal_pdf(x: np.Array, mu: float, cov: np.Array):
        """
        Computes the pdf of a Multi Gaussian normal sample x.
        parameters
        ----------
        x: np array, the sample
        mu: the expected value of x
        cov: the covariance matrix of x
        """

        d: float = len(x)
        exponent: float = np.dot(
            np.dot(np.transpose(-(1 / 2) * (x - mu))), np.linalg.inv(cov),
            x - mu)[0]
        return 1 / (np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(cov))) * \
               np.power(np.e, exponent)

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        normal_pdf_vector: np.Array = \
            np.vectorize(MultivariateGaussian._multi_normal_pdf)
        return normal_pdf_vector(X, self.mu_, self.cov_)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        dot_sum: np.Array = np.sum((X - mu) @ np.linalg.inv(cov) * (X - mu))
        return -(1/2) * (
                len(X) * np.log(
                                2 * np.power(np.pi, len(X[0])
                                )
                                * np.linalg.det(cov))
                + dot_sum)
