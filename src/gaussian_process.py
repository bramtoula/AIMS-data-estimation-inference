import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# TODO Check proper names of all params


class GaussianProcess:
    """Instance of a gaussian process
    """

    def __init__(self, covariance_function='rbf', var_meas_noise=0.01, kernel_params=[]):
        """Constructor of GaussianProcess

        Args:
            covariance_function (str, optional): String defining the covariance function. Includes a combination of 'rbf', 'rat_quad', and 'periodic' linked by '+' or '*'. All terms and operators must be separated by a space. Example: 'rbf + rat_quad * periodic'. Defaults to 'rbf'.
            var_meas_noise (float, optional): Variance of the measurement noise. Defaults to 0.01.
            kernel_params (list of tuples, optional): List of tuples where each tuple has parameters of a kernel term. Defaults to [].
        """
        self.covariance_computer = CovarianceComputer(
            covariance_function, kernel_params)
        self.var_meas_noise = var_meas_noise

    def set_covariance_params_manually(self, kernel_params, kernel_multipliers=[]):
        """Set parameters of covariance function manually

        Args:
            kernel_params (list of tuples): List of tuples where each tuple has parameters of a kernel term. 
            kernel_multipliers (list of floats, optional): Floats that are multiplied to each term of the covariance function. Defaults to [].
        """
        self.covariance_computer.update_params(
            kernel_params, kernel_multipliers)

    def set_covariance_params_optimize(self, t_train, y_train, init_kernel_params, bounds_kernel_params, init_multipliers=[], bounds_multipliers=[]):
        """Set parameters of covariance by optimisation to minimize the negative log likelihood. Requires initial values and bounds.

        Args:
            t_train (1D numpy array): Times of readings used for fitting model.
            y_train (1D numpy array): Sensor readings used for fitting model.
            init_kernel_params (list of tuples): List of tuples where each tuple has parameters of a kernel term.
            bounds_kernel_params (list of list of tuples): List of list of tuples where each list gives tuples of bounds for parameters of each kernel term. Each tuple of bounds has a minimum and maximum, and None can be used to ignore a bound.
            init_multipliers (list, optional): Initial floats that are multiplied to each term of the covariance function. . Defaults to [].
            bounds_multipliers (list of tuples, optional): List of tuple where each tuple contains minimum and maximum value for a multiplier of a term of the covariance function. Defaults to [].

        Returns:
            OptimizeResult: Optimization results
        """

        # Keep information about number of parameters for each term of the covariance function
        block_sizes_kernel_params = [len(params)
                                     for params in init_kernel_params]

        # Flatten kernel parameters and multipliers
        flat_params = [
            item for sublist in init_kernel_params for item in sublist] + init_multipliers

        # Run minimization of negative marginal log likelihood
        res = minimize(self.get_nll_kernel, flat_params, args=(block_sizes_kernel_params, t_train,
                                                               y_train), options={'disp': True}, bounds=bounds_kernel_params + bounds_multipliers)

        res_flatten_params_and_multipliers = res.x
        res_kernel_params = []

        # Deflatten params in results
        for block_size in block_sizes_kernel_params:
            res_kernel_params.append(
                res_flatten_params_and_multipliers[:block_size])
            res_flatten_params_and_multipliers = res_flatten_params_and_multipliers[block_size:]

        # After removing all kernel params, only the multipliers are left
        res_multipliers = res_flatten_params_and_multipliers

        # Update parameters of the covariance computer with optimization results
        self.covariance_computer.update_params(
            res_kernel_params, res_multipliers)
        return res

    def get_nll_kernel(self, kernel_and_multiplier_params, block_sizes_kernel_params, t_train, y_train):
        """Get negative marginal log likelihood of gaussian process given parameters of covariance functinon and training data. 

        Args:
            kernel_and_multiplier_params (list): flattened list containing all parameters and multipliers of covariance function
            block_sizes_kernel_params (list): list containing the number of parameters for each term of the covariance function
            t_train (1D numpy array): Times of readings used for fitting model.
            y_train (1D numpy array): Sensor readings used for fitting model.

        Returns:
            float: Negative marginal log likelihood
        """

        # Parse flattened kernel params
        kernel_params = []
        for block_size in block_sizes_kernel_params:
            kernel_params.append(kernel_and_multiplier_params[:block_size])
            kernel_and_multiplier_params = kernel_and_multiplier_params[block_size:]

        # After removing all kernel params, only the multipliers are left
        kernel_multipliers = kernel_and_multiplier_params

        self.covariance_computer.update_params(
            kernel_params, kernel_multipliers)

        # Use t_train at the end just to fill in data since it does not affect log marginal likelihood.
        _, _, _, _, log_marg_likelihood = self.fit_to_data(
            t_train, y_train, t_train)

        return -log_marg_likelihood

    # Model training
    def fit_to_data(self, x_input, y_target, x_test):
        """Fit Gaussian Process to data and evaluate predictive posterior distribution

        Args:
            x_input (1D numpy array): Times of readings used for fitting model.
            y_target (1D numpy array): Sensor readings used for fitting model.
            x_test (1D numpy array): Times at which to check predictive posterior distribution.

        Returns:
            [1D numpy array, 1D numpy array, 1D numpy array, 2D numpy array, float]: Elements returned are mean of posterior predictive destribution at testing points, variance of posterior predictive destribution at testing points, standard dev. of posterior predictive destribution at testing points, covariance of posterior predictive destribution, log marginal likelihood. 
        """
        n = len(x_input)
        K_train = self.covariance_computer.compute_covariance(x_input, x_input)
        K_test_train = self.covariance_computer.compute_covariance(
            x_test, x_input)
        K_test = self.covariance_computer.compute_covariance(x_test, x_test)
        L = np.linalg.cholesky(K_train+np.eye(n)*self.var_meas_noise)
        alpha = np.linalg.pinv(L.T)@(np.linalg.pinv(L)@y_target)

        pred_mean = K_test_train @ alpha

        v = np.linalg.pinv(L) @ K_test_train.T

        pred_cov = K_test - v.T @ v
        log_marg_likelihood = -0.5 * y_target.T @ alpha - \
            np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
        pred_var = np.diag(pred_cov)
        pred_sigma = np.sqrt(pred_var)
        return pred_mean.squeeze(), pred_var, pred_sigma, pred_cov, log_marg_likelihood.squeeze()


# Different kernel functions and their covariance computation
class RBFKernel:
    def __init__(self, params=(0.1, 0.3)):
        self.params = params
        self.sigma = params[0]
        self.length = params[1]
        self.type = 'rbf'

    def update_params(self, params):
        if len(params) != 2:
            print(
                'ERROR: Expected only 2 params to update RBF kernel params but got '+str(len(params)))
        self.params = params
        self.sigma = params[0]
        self.length = params[1]

    def compute_covariance(self, x1, x2):
        x_sq_dists = cdist(x1, x2, 'sqeuclidean')
        K = self.sigma**2 * np.exp(-(x_sq_dists)/(2*self.length*self.length))
        return K


class PeriodicKernel:
    def __init__(self, params=(0.5, 0.08)):
        self.params = params
        self.sigma = params[0]
        self.length = params[1]
        if len(params) == 3:
            self.period = params[2]
        # Default period taken from the real period of tides. Allow fo only two parameters to be given if this stays fixed.
        else:
            self.period = 12.42

        self.type = 'periodic'

    def update_params(self, params):
        if len(params) != 2 and len(params) != 3:
            print(
                'ERROR: Expected only 2 or 3 params to update Periodic kernel params but got '+str(len(params)))
        self.params = params
        self.sigma = params[0]
        self.length = params[1]
        if len(params) == 3:
            self.period = params[2]

    def compute_covariance(self, x1, x2):
        x_dists = cdist(x1, x2, 'euclidean')
        K = self.sigma**2 * \
            np.exp(-2/(self.length*self.length) *
                   np.sin(np.pi*x_dists/self.period)**2)
        return K


class RationalQuadraticKernel:
    def __init__(self, params=(0.2, 0.05, 0.001)):
        self.params = params
        self.sigma = params[0]
        self.alpha = params[1]
        self.length = params[2]

        self.type = 'rat_quad'

    def update_params(self, params):
        if len(params) != 3:
            print('ERROR: Expected only 3 params to update Rational Quadratic kernel params but got '+str(len(params)))
        self.params = params
        self.sigma = params[0]
        self.alpha = params[1]
        self.length = params[2]

    def compute_covariance(self, x1, x2):
        x_sq_dists = cdist(x1, x2, 'sqeuclidean')
        K = self.sigma**2 / \
            ((1+1/(2*self.alpha)*x_sq_dists/self.length**2)**self.alpha)
        return K


# Covariance computer that can easily combine kernels
class CovarianceComputer:
    """Object to build and compute covariance functions
    """

    def __init__(self, covariance_function, kernel_params=[], kernel_term_multipliers=[]):
        """Constructor of CovarianceComputer object

        Args:
            covariance_function (str, optional): String defining the covariance function. Includes a combination of 'rbf', 'rat_quad', and 'periodic' linked by '+' or '*'. All terms and operators must be separated by a space. Example: 'rbf + rat_quad * periodic'. Defaults to 'rbf'.
            kernel_params (list, optional): List of tuples where each tuple has parameters of a kernel term. Defaults to [].
            kernel_term_multipliers (list, optional): Floats that are multiplied to each term of the covariance function. Defaults to [].
        """
        self.kernel_ops = []
        self.kernels = []

        # Build list of kernel components and list of operations between them from covariance function string
        for s in covariance_function.split(' '):
            if s == '+' or s == '*':
                self.kernel_ops.append(s)
            elif s == 'rbf':
                # Check if kernel params are given
                if len(kernel_params) > len(self.kernels) and len(kernel_params[len(self.kernels)]) == 2:
                    self.kernels.append(
                        RBFKernel(kernel_params[len(self.kernels)]))
                else:
                    self.kernels.append(RBFKernel())
            elif s == 'periodic':
                # Check if kernel params are given
                if len(kernel_params) > len(self.kernels) and (len(kernel_params[len(self.kernels)]) == 2 or len(kernel_params[len(self.kernels)]) == 3):
                    self.kernels.append(PeriodicKernel(
                        kernel_params[len(self.kernels)]))
                else:
                    self.kernels.append(PeriodicKernel())
            elif s == 'rat_quad':
                # Check if kernel params are given
                if len(kernel_params) > len(self.kernels) and len(kernel_params[len(self.kernels)]) == 2:
                    self.kernels.append(RationalQuadraticKernel(
                        kernel_params[len(self.kernels)]))
                else:
                    self.kernels.append(RationalQuadraticKernel())
            else:
                print('ERROR: Could not identify covariance function string term -> '+s)

        if len(self.kernel_ops) != (len(self.kernels) - 1):
            print('ERROR: Invalid number of operations compared to kernel terms found')
            return

        # Parse multipliers to each kernel term
        # If no coefficient given, assume all are 1
        if len(kernel_term_multipliers) == 0:
            self.kernel_term_multipliers = [1] * len(self.kernels)
        elif len(kernel_term_multipliers) == len(self.kernels):
            self.kernel_term_multipliers = kernel_term_multipliers
        else:
            print(
                'ERROR: Received number of kernel term multipliers inconsistent with number of kernel terms')
            return

    def update_params(self, kernel_params, kernel_term_multipliers=[]):
        """Update parameters of kernel terms of the covariance function

        Args:
            kernel_params (list, optional): List of tuples where each tuple has parameters of a kernel term. Defaults to [].
            kernel_term_multipliers (list, optional): Floats that are multiplied to each term of the covariance function. Defaults to [].
        """
        if len(kernel_params) != len(self.kernels):
            print('ERROR: Trying to update kernel params, but number of params given is different from number of kernel terms')
            return

        # Update kernel parameters
        for i, k in enumerate(kernel_params):
            self.kernels[i].update_params(k)

        # Update kernel term multipliers
        if len(kernel_term_multipliers) > 0:
            if len(kernel_term_multipliers) == len(self.kernels):
                self.kernel_term_multipliers = kernel_term_multipliers
            else:
                print('ERROR: Trying to update kernel term multipliers, but number of coefficients given is different from number of kernel terms')
                return

    def compute_covariance(self, x1, x2):
        """Computer covariance function for two sets of points

        Args:
            x1 (1D numpy array)
            x2 (1D numpy array)

        Returns:
            2D numpy array: Resulting covariance
        """

        # Compute covariances for each term in the function and apply multipliers
        covariances = [self.kernel_term_multipliers[i] *
                       self.kernels[i].compute_covariance(x1, x2) for i in range(len(self.kernels))]

        # Go through multiplications between covariances first.
        cov_ids_to_remove = []
        for i, op in enumerate(self.kernel_ops):
            if op == '*':
                covariances[i +
                            1] = np.multiply(covariances[i], covariances[i+1])
                cov_ids_to_remove.append(i)

        # Remove covariances which have already been added to another element by a multiplication
        for index in sorted(cov_ids_to_remove, reverse=True):
            del covariances[index]

        # Sum up the remaining terms (only operations remaining must be additions)
        return np.sum(covariances, axis=0)

    def add_to_kernel_param(self, kernel_term_id, param_to_add_id, amount_to_add):
        """Add a constant to a parameter of one term of the covariance function.

        Args:
            kernel_term_id (int): ID of the term if which a parameter should be modified
            param_to_add_id (int): ID of the parameter of the term that should be modified
            amount_to_add (float): Amount to add to selected parameter
        """
        cur_params = self.kernels[kernel_term_id].params
        cur_params[param_to_add_id] += amount_to_add
        self.kernels[kernel_term_id].update_params(cur_params)
