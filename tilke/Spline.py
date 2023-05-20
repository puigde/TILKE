from scipy.interpolate import splrep, splev
import numpy as np
import matplotlib.pyplot as plt
import math
import noise


class Spline:
    """Custom Spline class for curve representation

    Attributes:
        spline (scipy spline) : inner scipy param to handle derivatives
                Default is None
        diff (scipy spline param) : inner scipy param to handle derivatives
            Default is None
        xt (list) : list of x coordinates
        yt (list) : list of y coordinates
        true_cones (list) : list of true cones
        true_cones_contaminated (list) : list of noisy cones
        data (list) : list of points (x,y) to interpolate
            Default is None
        t (list) : parameter space for the curve
            Default is None
        highlighted_points (list) : list of highlighted points

    Credits to Albert Rodas from BCN eMotorsport autonomous controls
    """

    def __init__(self, t=None, data=None, s=0, spline=None, diff=0, k=3):
        """Init method for the Spline class

        Arguments:
            t (list) : parameter space for the curve
                Default is None
            data (list) : list of points (x,y) to interpolate
                Default is None
            s (float) : smoothing factor
            spline (scipy spline) : inner scipy param to handle derivatives
                Default is None
            diff (scipy spline param) : inner scipy param to handle derivatives
                Default is None
            k (int) : the degree of the Spline fit
        """
        self.spline = spline
        self.diff = diff
        self.t = t
        self.highlighted_points = []
        self.true_cones = []
        self.true_cones_contaminated = []
        self.data = data
        data = np.array(data)
        if self.t is None:
            self._get_t_acum_distance(data=data)
        if spline is None:
            self.xt = splrep(x=self.t, y=data[0], s=s, per=1, k=k)
            self.yt = splrep(x=self.t, y=data[1], s=s, per=1, k=k)
            self.spline = [self.xt, self.yt]
        else:
            self.xt = spline[0]
            self.yt = spline[1]

    def __call__(self, t):
        """Call method for the Spline class allows to evaluate a Spline at a certain t

        Arguments:
            t (float) : points of the parameter where the spline is evaluated
                Default is None
        Returns:
            np.array : array containings to the evaluation of the spline
        """
        gamma = np.array(
            [splev(t, self.xt, der=self.diff), splev(t, self.yt, der=self.diff)]
        ).transpose()
        return gamma

    def __getitem__(self, t):
        """Call method for the Spline class allows to evaluate a Spline at a certain t

        Arguments:
            t (float) : points of the parameter where the spline is evaluated
                Default is None
        Returns:
            np.array : array containings to the evaluation of the spline
        """
        gamma = np.array(
            [splev(t, self.xt, der=self.diff), splev(t, self.yt, der=self.diff)]
        ).transpose()
        return gamma

    def derivative(self):
        """Obtains a new Spline which is the derivative of the actual one"""
        dgamma = Spline(data=self.data, diff=self.diff + 1)
        return dgamma

    def plot(self, precision: int = 1000, showcones: bool = True):
        """Plots a given spline with it"s highlighted points and true cones

        Arguments:
            precision (int) the precision to gen the points in eval interval
                Default is 1000
            showcones (bool) if true shows the true cones
                Default is True
        """
        tt = np.linspace(self.t[0], self.t[-1], precision)
        gamma = self(tt)
        plt.plot(gamma[:, 0], gamma[:, 1])
        if len(self.true_cones) > 0 and showcones:
            hc = np.array([self(c) for c in self.true_cones])
            plt.scatter(hc[:, 0], hc[:, 1], c="y")
        if len(self.highlighted_points) > 0:
            hp = np.array([self(t) for t in self.highlighted_points])
            plt.scatter(hp[:, 0], hp[:, 1], c="y")

    def _get_t_acum_distance(self, data):
        """Computes the accumulated dinstance in between the points in the spline
        and stores it in the parameter t

        Arguments:
            data (list) : list of points (x,y) to interpolate
        """
        sq_diffvec = np.square(np.diff(data))
        t_end = np.cumsum(np.sqrt(sq_diffvec[0] + sq_diffvec[1]))
        self.t = np.append([0], t_end)

    def add_highlighted_points(self, tlist: list):
        """Adds the evaluation of the spline at a time t as a highlighted point which will be included in plot

        Arguments:
            tlist (list) : list of t for points to be highlighted
        """
        if isinstance(tlist, int) or isinstance(tlist, float):
            tlist = [tlist]
        for t in tlist:
            self.highlighted_points.append(t)

    def add_true_cone(self, c):
        """Adds true cone to the spline layout

        Arguments:
            c (float) : t value of the true cone
        """
        self.true_cones.append(c)

    def get_int_ext_splines(
        self,
        dist: float = 2,
        stepsize: float = 0.5,
        smoothing: float = 0.5,
        sampling_factor: int = 5,
    ):
        """Computes the interior and exterior splines of the given spline at a given distance

        Arguments:
            dist (float) : distance to the interior and exterior splines
                Default is 2
            stepsize (float) : stepsize to compute the interior and exterior splines
                Default is 0.5
            smoothing (float) : smoothing factor for the interior and exterior splines
                Default is 0.5
            sampling_factor (int) : sampling factor for the interior and exterior splines
                Default is 5
        Returns:
            tuple : tuple containing:
                interior_spline (Spline) : interior spline
                exterior_spline (Spline) : exterior spline
        """
        t = self.t
        curvegen_steps = np.linspace(t[0], t[-1], math.floor((t[-1]) / stepsize))
        x_int = []
        x_ext = []
        y_int = []
        y_ext = []
        d = self.derivative()
        for t in curvegen_steps:
            x, y = self[t]
            dx, dy = d[t]
            dmulti = dist / np.sqrt(dx**2 + dy**2)
            x_int += [x + dmulti * dy]
            y_int += [y - dmulti * dx]
            x_ext += [x - dmulti * dy]
            y_ext += [y + dmulti * dx]

        interior_data_sample = [
            list(x_int[::sampling_factor]),
            list(y_int[::sampling_factor]),
        ]
        exterior_data_sample = [
            list(x_ext[::sampling_factor]),
            list(y_ext[::sampling_factor]),
        ]
        interior_spline = Spline(data=interior_data_sample, s=smoothing)
        exterior_spline = Spline(data=exterior_data_sample, s=smoothing)
        return interior_spline, exterior_spline

    def get_int_ext_splines_perlin_noise(
        self,
        dist: float = 2,
        stepsize: float = 0.5,
        smoothing: float = 0.5,
        sampling_factor: int = 5,
    ):
        """Computes the interior and exterior splines of the given spline at a given distance with perlin noise

        Arguments:
            dist (float) : minimum distance to the interior and exterior splines
                Default is 2
            stepsize (float) : stepsize to compute the interior and exterior splines
                Default is 0.5
            smoothing (float) : smoothing factor for the interior and exterior splines
                Default is 0.5
            sampling_factor (int) : sampling factor for the interior and exterior splines
                Default is 5
        Returns:
            tuple : tuple containing:
                interior_spline (Spline) : interior spline
                exterior_spline (Spline) : exterior spline
        """
        t = self.t
        curvegen_steps = np.linspace(t[0], t[-1], math.floor((t[-1]) / stepsize))
        x_int = []
        x_ext = []
        y_int = []
        y_ext = []
        d = self.derivative()
        for t in curvegen_steps:
            x, y = self[t]
            dx, dy = d[t]
            dmulti = (dist + noise.pnoise1(t)) / np.sqrt(dx**2 + dy**2)
            x_int += [x + dmulti * dy]
            y_int += [y - dmulti * dx]
            x_ext += [x - dmulti * dy]
            y_ext += [y + dmulti * dx]
        interior_data_sample = [
            list(x_int[::sampling_factor]),
            list(y_int[::sampling_factor]),
        ]
        exterior_data_sample = [
            list(x_ext[::sampling_factor]),
            list(y_ext[::sampling_factor]),
        ]
        interior_spline = Spline(data=interior_data_sample, s=smoothing)
        exterior_spline = Spline(data=exterior_data_sample, s=smoothing)
        return interior_spline, exterior_spline
