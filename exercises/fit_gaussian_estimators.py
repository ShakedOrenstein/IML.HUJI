from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu: int = 10
    samples: np.array = np.random.normal(mu, 1, 1000)
    uni_gaus: UnivariateGaussian = UnivariateGaussian()
    uni_gaus.fit(samples)
    print('(' + str(uni_gaus.mu_) + ', ' + str(uni_gaus.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent
    samples_num: np.array = np.arange(10, len(samples) + 1, 10)
    absolute_dist = []
    for i in samples_num:
        gaus_i = UnivariateGaussian()
        gaus_i.fit(samples[:i])
        absolute_dist.append(np.abs(mu - gaus_i.mu_))

    distance_fig = go.Figure([go.Scatter(x=samples_num, y=absolute_dist,
                                mode='markers+lines', name='absolute distance')],
                    layout=go.Layout(title='Absolute Distance Between Estimation'
                                           'and the True Value of Expectation as a'
                                           'Function of Number of Samples',
                                     xaxis_title="Absolute Distance",
                                     yaxis_title="Number of Samples",
                                     height=300))
    distance_fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    samples_pdfs = uni_gaus.pdf(samples)
    pdf_fig = go.Figure([go.Scatter(x=samples, y=samples_pdfs,
                                mode='markers',
                                name='Empirical PDF')],
                    layout=go.Layout(
                        title='Empirical PDF of the Samples',
                        xaxis_title="Sample Values",
                        yaxis_title="PDF",
                        height=300))
    pdf_fig.show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples: np.array = np.random.multivariate_normal(mu, cov, 1000)
    multi_gaus: MultivariateGaussian = MultivariateGaussian()
    multi_gaus.fit(samples)
    print("Expectation: " + str(multi_gaus.mu_))
    print("Covariance matrix: " + str(multi_gaus.cov_))

    # Question 5 - Likelihood evaluation
    axis_values: np.Array = np.linspace(-10, 10, 200)
    log_likelihood_values = []
    max_f1: float = axis_values[0]
    max_f3: float = axis_values[0]
    max_log_likelihood: float = MultivariateGaussian.log_likelihood(np.array([max_f1, 0, max_f3, 0]), cov, samples)
    for i in range(len(axis_values)):
        for j in range(len(axis_values)):
            temp_mu: np.Array = np.array([axis_values[i], 0, axis_values[j], 0])
            temp_log_likelihood: float = MultivariateGaussian.log_likelihood(temp_mu, cov, samples)
            log_likelihood_values.append(temp_log_likelihood)
            if temp_log_likelihood > max_log_likelihood:
                max_log_likelihood = temp_log_likelihood
                max_f1 = axis_values[i]
                max_f3 = axis_values[j]

    log_likelihood_values = np.array(log_likelihood_values).reshape(len(axis_values), len(axis_values))

    go.Figure(go.Heatmap(x=axis_values, y=axis_values, z=log_likelihood_values))\
        .update_layout(title='Multi Gaussian Log-Likelihood as a Function of the Expectation')\
        .update_xaxes(title='f1')\
        .update_yaxes(title='f3').show()

    # Question 6 - Maximum likelihood
    print("Max f1: " + str(round(max_f1, 3)))
    print("Max f3: " + str(round(max_f3, 3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
