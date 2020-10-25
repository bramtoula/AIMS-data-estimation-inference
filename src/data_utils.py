
import pandas as pd
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np


def load_data_from_csv(file_path):
    return pd.read_csv(file_path, parse_dates=[0, 2])


def datetime_to_hours_passed(datetime_series):
    """Takes in a panda series of datetimes and returns a series with hours passed since the first reading.

    Args:
        panda series: series with datetimes

    Returns:
        panda series: series with hours since first reading
    """
    hours = (datetime_series-datetime_series[0]).dt.total_seconds()/3600
    return hours


def center_readings(readings):
    mean = readings.mean()
    return readings - mean, mean


class Visualizer:
    """Class to handle plotting of results
    """

    def __init__(self, t_sensor_readings, y_sensor_readings, t_test, y_gt_at_test):
        """Initialize visualizer object

        Args:
            t_sensor_readings (list or 1D numpy array): Time of sensor readings used for training model
            y_sensor_readings (list or 1D numpy array): Values of sensor readings used for training model
            t_test (list or 1D numpy array): Times at which ground truth is given for evaluation
            y_gt_at_test (list or 1D numpy array): Values of ground truth for evaluation
        """
        self.t_sensor_readings = t_sensor_readings
        self.y_sensor_readings = y_sensor_readings
        self.t_test = t_test
        self.y_gt_at_test = y_gt_at_test


    def plot(self, t_pred, pred_mean, pred_sigma, pred_cov, log_marg_likelihood, nb_draws=5, title = None, save_path = None):
        """Generate plot to visualize GP estimates

        Args:
            t_pred (list or 1D numpy array): Times at which predictions are given
            pred_mean (list or 1D numpy array): Predicted mean values of the posterior predictive distribution
            pred_sigma (list or 1D numpy array): Predicted standard deviation values of the posterior predictive distribution
            pred_cov (2D numpy array): Covariance matrix of posterior predictive distribution
            log_marg_likelihood (float): Log of marginal likelihood
            nb_draws (int, optional): Number of function draws to plot. Defaults to 5.
            title (string, optional): Title to put on plot. Defaults to None.
            save_dir (string, optional): Path in which to save image. Defaults to None.
        """
        fig = plt.figure()
        ax = plt.gca()
        plt.xlabel('Hours since first reading')
        plt.ylabel('Tide height difference to mean (m)')
        plt.ylim([-2.5, 2.5])

        if title is not None:
            plt.title(title)

        # Sensor readings and groundtruth
        plt.scatter(self.t_sensor_readings, self.y_sensor_readings,
                    5, label='Sensor readings', color='blue')
        plt.scatter(self.t_test, self.y_gt_at_test, 5,
                    label='Ground truth', color='c')

        # Posterior predictive distribution visualization with mean and standard deviations
        plt.plot(t_pred, pred_mean, label='Posterior predicted mean',
                 color='red', alpha=0.8, linewidth=1)
        plt.fill_between(t_pred.squeeze(), pred_mean-pred_sigma, pred_mean +
                         pred_sigma, label='1 $\sigma$', alpha=0.5, color='orange')

        plt.fill_between(t_pred.squeeze(), pred_mean-2*pred_sigma, pred_mean +
                         2*pred_sigma, label='2 $\sigma$', alpha=0.3, color='orange')

        # Draw samples from predicted distribution
        if nb_draws > 0:
            draws = np.random.multivariate_normal(
                pred_mean, pred_cov, nb_draws)
            plt.plot(t_pred, draws[0], label='Drawn from distribution',
                     color='k', alpha=1, linewidth=0.2)
            for d in draws[1:]:
                plt.plot(t_pred, d, color='black', alpha=1, linewidth=0.2)

        # Write log marginal likelihood on plot
        plt.text(0.4, 0.94, 'Log marginal likelihood: '+'{0:.2f}'.format(
            log_marg_likelihood), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


        plt.legend(loc='lower left',ncol=3)

        if save_path is not None:
            plt.savefig(save_path)