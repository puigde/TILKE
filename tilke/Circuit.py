from tilke.Spline import Spline
import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
from matplotlib.patches import Polygon
from scipy.spatial import KDTree
import noise
import random
import tilke.image_utils as image_utils
from datetime import datetime
import csv
from tilke.CurveGenerator import BezierCurveGenerator
from dataclasses import dataclass


@dataclass
class CircuitRestrictions:
    """Parameters for the circuit generation

    Attributes:
        _interior_curve_curvature (bool): Whether the interior curve curvature satisfies the restrictions
            Default: False
        _exterior_curve_curvature (bool): Whether the exterior curve curvature satisfies the restrictions
            Default: False
        _middle_curve_curvature (bool): Whether the middle curve curvature satisfies the restrictions
            Default: False
        _interior_exterior_distance (bool): Whether the interior and exterior curves distance satisfies the restrictions
            Default: False
        _interior_middle_distance (bool): Whether the interior and middle curves distance satisfies the restrictions
            Default: False
        _exterior_middle_distance (bool): Whether the exterior and middle curves distance satisfies the restrictions
            Default: False
        min_curvature_radius_middle_curve (float): The minimum curvature radius for the middle curve
            Default: 6.5
        min_curvature_radius_interior_curve (float): The minimum curvature radius for the interior curve
            Default: 4.5
        min_curvature_radius_exterior_curve (float): The minimum curvature radius for the exterior curve
            Default: 8
        stepsize_interior_curvature_checking (float): The stepsize for checking the interior curve curvature
            Default: 1.0
        stepsize_exterior_curvature_checking (float): The stepsize for checking the exterior curve curvature
            Default: 1.0
        stepsize_middle_curvature_checking (float): The stepsize for checking the middle curve curvature
            Default: 1.0
        stepsize_middle_distance_checking (float): The stepsize for checking the middle curve distance
            Default: 1.0
        stepsize_interior_distance_checking (float): The stepsize for checking the interior curve distance
            Default: 1.0
    """

    _interior_curve_curvature: bool = False
    _exterior_curve_curvature: bool = False
    _middle_curve_curvature: bool = False
    _interior_exterior_distance: bool = False
    _interior_middle_distance: bool = False
    _exterior_middle_distance: bool = False
    min_curvature_radius_middle_curve: float = 6.5
    min_curvature_radius_interior_curve: float = 4.5
    min_curvature_radius_exterior_curve: float = 8
    stepsize_interior_curvature_checking: float = 1.0
    stepsize_exterior_curvature_checking: float = 1.0
    stepsize_middle_curvature_checking: float = 1.0
    stepsize_middle_distance_checking: float = 1.0
    stepsize_interior_distance_checking: float = 1.0
    stepsize_exterior_distance_checking: float = 1.0
    min_interior_exterior_distance: float = 1.0
    min_interior_middle_distance: float = 1.0
    min_exterior_middle_distance: float = 1.0


@dataclass
class ConePopulationParameters:
    """Parameters for the cone population generation

    Attributes:
        total_number_cones_lower_bound (int): lower bound for the total number of cones
            Default is 90
        total_number_cones_upper_bound (int): upper bound for the total number of cones
            Default is 130
        minimum_proportion_cones_per_curve (float): minimum proportion of cones per curve
            Default is 0.4
        total_number_cones (int): total number of cones
            Default is 130
        false_negative_probability (float): probability of a cone not being detected
            Default is 0.1
        false_positive_probability (float): probability of a cone being detected when there is no cone
            Default is 0.07
        interior_curve_range_boost (float): boost miss probability for adjacent cones for the interior curve
            Default is 0.025
        exterior_curve_range_boost (float): bost miss probability for adjacent cones for the exterior curve
            Default is 0.025
    """

    total_number_cones_lower_bound: int = 90
    total_number_cones_upper_bound: int = 130
    minimum_proportion_cones_per_curve: float = 0.4
    total_number_cones: int = 130
    false_negative_probability: float = 0.1
    false_positive_probability: float = 0.07
    interior_curve_range_boost: float = 0.025
    exterior_curve_range_boost: float = 0.025


@dataclass
class TrackParameters:
    """Parameters for the circuit generation

    Attributes:
        distance_between_middle_and_track_sides (float): The distance between the middle curve and the track sides
            Default: 2
        stepsize_for_track_generation (float): Interval stepsize for generating the track
            Default: 0.5
        smoothing_for_track_generation (float): Spline smoothing for generating the track
            Default: 0.5
        sampling_factor_for_track_generation (int): Spline generation has a lot of points and could crash some parts so we sample the points
            Default: 5
    """

    distance_between_middle_and_track_sides: float = 2
    stepsize_for_track_generation: float = 0.5
    smoothing_for_track_generation: float = 0.5
    sampling_factor_for_track_generation: int = 5


class Circuit:
    """Generates a randomized closed smooth circuit layout.

    Attributes:
        interior_curve (Spline) : Representation of the interior curve of the circuit
        exterior_curve (Spline) : Representation of the exterior curve of the circuit
        middle_curve (Spline) : Representation of a middle curve in between the interior and exterior curves
        false_cones (list) : list of cones that are not valid
        circuit_restrictions (CircuitRestrictions) : parameters for the circuit generation
        cone_population_parameters (ConePopulationParameters) : parameters for the cone population generation
        restrictions_compliant (bool) : whether the circuit to generate should follow the defined curvature and distance restriction parameters
        orientation (str): orientation of the circuit, either clockwise, counter_clockwise or random
    """

    def __init__(
        self,
        middle_curve: Spline = None,
        restrictions_compliant: bool = True,
        orientation: str = "random",
        seed: int = None,
        curve_generator: BezierCurveGenerator = None,
        circuit_restrictions: CircuitRestrictions = None,
        cone_population_parameters: ConePopulationParameters = None,
        track_parameters: TrackParameters = None,
    ):
        """Init method for the Circuit class

        Arguments:
            middle_curve (Spline) : Representation of a middle curve in between the interior and exterior curves
            restrictions_compliant (bool) : whether the circuit to generate should follow the defined curvature and distance restriction parameters
            orientation (str): orientation of the circuit, either clockwise, counter_clockwise or random
            seed (int): seed for the random number generator
            curve_generator (BezierCurveGenerator): generator for the middle curve
            circuit_restrictions (CircuitRestrictions): parameters for the circuit generation
            cone_population_parameters (ConePopulationParameters): parameters for the cone population generation
            track_parameters (TrackParameters): parameters for generating the interior and exterior curves from the middle curve
        """
        self.restrictions_compliant = restrictions_compliant
        self.orientation = orientation
        if circuit_restrictions is None:
            self.circuit_restrictions = CircuitRestrictions()
        else:
            self.circuit_restrictions = circuit_restrictions
        if curve_generator is None:
            self.curve_generator = BezierCurveGenerator(seed=seed)
        else:
            self.curve_generator = curve_generator
        if cone_population_parameters is None:
            self.cone_population_parameters = ConePopulationParameters()
        else:
            self.cone_population_parameters = cone_population_parameters
        if track_parameters is None:
            self.track_parameters = TrackParameters()
        else:
            self.track_parameters = track_parameters
        self.false_cones = []
        while True:
            if middle_curve is None:
                self.middle_curve = self.curve_generator.run()
            else:
                self.middle_curve = middle_curve

            (
                self.interior_curve,
                self.exterior_curve,
            ) = self.middle_curve.get_int_ext_splines(
                dist=self.track_parameters.distance_between_middle_and_track_sides,
                stepsize=self.track_parameters.stepsize_for_track_generation,
                smoothing=self.track_parameters.smoothing_for_track_generation,
                sampling_factor=self.track_parameters.sampling_factor_for_track_generation,
            )
            if self.restrictions_compliant:
                if self.check_validity_curvature() and self.check_validity_distances():
                    break
                else:
                    if seed is not None or middle_curve is not None:
                        self.restrictions_compliant = False
                        return
                    middle_curve = None

        if orientation == "counter_clockwise" or (
            orientation == "random" and random.random() > 0.5
        ):
            self.interior_curve = Spline(
                data=self.interior_curve.data[::-1],
                s=self.curve_generator.smoothing_factor,
            )
            self.exterior_curve = Spline(
                data=self.exterior_curve.data[::-1],
                s=self.curve_generator.smoothing_factor,
            )
            self.middle_curve = Spline(
                data=self.middle_curve.data[::-1],
                s=self.curve_generator.smoothing_factor,
            )
            self.orientation = "counter_clockwise"
        elif orientation == "clockwise":
            self.orientation = "clockwise"

    def check_validity_curvature(self) -> bool:
        """Checks the validity of the curvature for the three curves in the circuit"""
        curvature_parameters = {
            "middle": {
                "curve": self.middle_curve,
                "min": self.circuit_restrictions.min_curvature_radius_middle_curve,
                "restrictions_key": "_middle_curve_curvature",
                "stepsize": self.circuit_restrictions.stepsize_middle_curvature_checking,
            },
            "interior": {
                "curve": self.interior_curve,
                "min": self.circuit_restrictions.min_curvature_radius_interior_curve,
                "restrictions_key": "_interior_curve_curvature",
                "stepsize": self.circuit_restrictions.stepsize_interior_curvature_checking,
            },
            "exterior": {
                "curve": self.exterior_curve,
                "min": self.circuit_restrictions.min_curvature_radius_exterior_curve,
                "restrictions_key": "_exterior_curve_curvature",
                "stepsize": self.circuit_restrictions.stepsize_exterior_curvature_checking,
            },
        }

        for id_curve in curvature_parameters:
            curve = curvature_parameters[id_curve]["curve"]
            eval_steps = np.linspace(
                curve.t[0],
                curve.t[-1],
                math.floor((curve.t[-1]) / curvature_parameters[id_curve]["stepsize"]),
            )
            curve_radius = lambda d, d2: (
                ((d[0] ** 2 + d[1] ** 2) ** (3 / 2))
                / np.abs(d[0] * d2[1] - d[1] * d2[0])
            )
            dd = curve.derivative()
            dd2 = dd.derivative()
            c_rad = np.array([curve_radius(dd(t), dd2(t)) for t in eval_steps])
            failed_steps = eval_steps[c_rad < curvature_parameters[id_curve]["min"]]
            curve.add_highlighted_points(failed_steps)
            setattr(
                self.circuit_restrictions,
                curvature_parameters[id_curve]["restrictions_key"],
                len(failed_steps) == 0,
            )
        return (
            self.circuit_restrictions._middle_curve_curvature
            and self.circuit_restrictions._interior_curve_curvature
            and self.circuit_restrictions._exterior_curve_curvature
        )

    def check_validity_distances(self):
        """Checks the validity of the distances between the three curves in the circuit
        uses a kdtree to find the closest points and checks if their distance is less than mindist to discard the circuit
        """
        middle_curve_steps = np.linspace(
            self.middle_curve.t[0],
            self.middle_curve.t[-1],
            math.floor(
                self.middle_curve.t[-1]
                / self.circuit_restrictions.stepsize_middle_distance_checking
            ),
        )
        middle_curve_points = self.middle_curve(middle_curve_steps)
        interior_curve_steps = np.linspace(
            self.interior_curve.t[0],
            self.interior_curve.t[-1],
            math.floor(
                self.interior_curve.t[-1]
                / self.circuit_restrictions.stepsize_interior_distance_checking
            ),
        )
        interior_curve_points = self.interior_curve(interior_curve_steps)
        exterior_curve_steps = np.linspace(
            self.exterior_curve.t[0],
            self.exterior_curve.t[-1],
            math.floor(
                self.exterior_curve.t[-1]
                / self.circuit_restrictions.stepsize_exterior_distance_checking
            ),
        )
        exterior_curve_points = self.exterior_curve(exterior_curve_steps)
        inteior_tree = KDTree(interior_curve_points)
        exterior_tree = KDTree(exterior_curve_points)
        int_ext_dist, _ = inteior_tree.query(exterior_curve_points)
        int_mid_dist, _ = inteior_tree.query(middle_curve_points)
        ext_mid_dist, _ = exterior_tree.query(middle_curve_points)
        self.circuit_restrictions._interior_exterior_distance = np.all(
            int_ext_dist > self.circuit_restrictions.min_interior_exterior_distance
        )
        self.circuit_restrictions._interior_middle_distance = np.all(
            int_mid_dist > self.circuit_restrictions.min_interior_middle_distance
        )
        self.circuit_restrictions._exterior_middle_distance = np.all(
            ext_mid_dist > self.circuit_restrictions.min_exterior_middle_distance
        )

        return (
            self.circuit_restrictions._interior_exterior_distance
            and self.circuit_restrictions._interior_middle_distance
            and self.circuit_restrictions._exterior_middle_distance
        )

    def populate_circuit_naive(self, n_cones: int = 130):
        """Populates the circuit with cones by adding them to the interior and exterior curves

        Arguments:
            n_cones (int) : total number of cones to populate the circuit with
        """
        upscale = self.exterior_curve.t[-1] / self.interior_curve.t[-1]
        n_interior_cones = n_cones // 2
        cone_positions = np.linspace(0, self.interior_curve.t[-1], n_interior_cones)
        for c in cone_positions:
            self.interior_curve.add_true_cone(c)
            self.exterior_curve.add_true_cone(c * upscale)

    def populate_circuit(self):
        """Populates the circuit with cones by adding them to the interior and exterior curves with
        an added perlin noise
        """
        cones = self.cone_population_parameters.total_number_cones
        upscale = self.exterior_curve.t[-1] / self.interior_curve.t[-1]
        n_interior_cones = cones // 2
        cone_positions = np.linspace(0, self.interior_curve.t[-1], n_interior_cones)
        for i, value in enumerate(cone_positions):
            cone_positions[i] += noise.pnoise1(value)
            if cone_positions[i] > self.interior_curve.t[-1]:
                cone_positions[i] = self.interior_curve.t[-1]
                break
        for c in cone_positions:
            self.interior_curve.add_true_cone(c)
            self.exterior_curve.add_true_cone(c * upscale)

    def populate_circuit_random_number_cones(self):
        """Populates the circuit with cones by adding them to the interior and exterior curves with
        a random number of cones
        """
        total_number_cones = random.randint(
            self.cone_population_parameters.total_number_cones_lower_bound,
            self.cone_population_parameters.total_number_cones_upper_bound + 1,
        )
        minimum_number_cones_per_curve = int(
            self.cone_population_parameters.minimum_proportion_cones_per_curve
            * total_number_cones
        )
        number_interior_cones = random.randint(
            minimum_number_cones_per_curve,
            total_number_cones - minimum_number_cones_per_curve + 1,
        )
        number_exterior_cones = total_number_cones - number_interior_cones
        interior_cone_positions = np.linspace(
            self.interior_curve.t[0],
            self.interior_curve.t[-1],
            number_interior_cones,
        )
        exterior_cone_positions = np.linspace(
            self.exterior_curve.t[0],
            self.exterior_curve.t[-1],
            number_exterior_cones,
        )
        for i, value in enumerate(interior_cone_positions):
            interior_cone_positions[i] += noise.pnoise1(value)
            if interior_cone_positions[i] > self.interior_curve.t[-1]:
                interior_cone_positions[i] = self.interior_curve.t[-1]
                break
        for i, value in enumerate(exterior_cone_positions):
            exterior_cone_positions[i] += noise.pnoise1(value)
            if exterior_cone_positions[i] > self.exterior_curve.t[-1]:
                exterior_cone_positions[i] = self.exterior_curve.t[-1]
                break
        for c in interior_cone_positions:
            self.interior_curve.add_true_cone(c)
        for c in exterior_cone_positions:
            self.exterior_curve.add_true_cone(c)

    def contaminate_cone_population(self):
        """Contaminates the cone population of the circuit by adding missed and false cones
        Arguments:
            mode (str) : the mode of contamination to use, the options are "BCNeMotorsport", "performance" and "noisy"
                Default: "BCNeMotorsport"
        """
        selected_indices_interior = []
        selected_indices_exterior = []
        for i in range(len(self.interior_curve.true_cones)):
            if (
                random.random()
                < self.cone_population_parameters.false_negative_probability
                + (
                    self.cone_population_parameters.interior_curve_range_boost
                    * (i - 1 in selected_indices_interior)
                )
            ):
                selected_indices_interior.append(i)
        for i in range(len(self.exterior_curve.true_cones)):
            if (
                random.random()
                < self.cone_population_parameters.false_negative_probability
                + (
                    self.cone_population_parameters.exterior_curve_range_boost
                    * (i - 1 in selected_indices_exterior)
                )
            ):
                selected_indices_exterior.append(i)
        self.interior_curve.true_cones_contaminated = [
            self.interior_curve.true_cones[i]
            for i in range(len(self.interior_curve.true_cones))
            if i not in selected_indices_interior
        ]
        self.exterior_curve.true_cones_contaminated = [
            self.exterior_curve.true_cones[i]
            for i in range(len(self.exterior_curve.true_cones))
            if i not in selected_indices_exterior
        ]
        sample_indices = np.linspace(0, self.exterior_curve.t[-1], 100)
        sample_points = [self.exterior_curve[i] for i in sample_indices]
        sample_points = np.array(sample_points)
        min_x = np.min(sample_points[:, 0])
        min_y = np.min(sample_points[:, 1])
        max_x = np.max(sample_points[:, 0])
        max_y = np.max(sample_points[:, 1])
        fp_x_extension = 80
        fp_y_extension = 80
        max_x += random.uniform(0, fp_x_extension)
        max_y += random.uniform(0, fp_y_extension)
        counter = 0
        for i in range(
            len(self.interior_curve.true_cones) + len(self.exterior_curve.true_cones)
        ):
            if (
                random.random()
                < self.cone_population_parameters.false_positive_probability
            ):
                counter += 1
                cc = [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
                self.false_cones.append(cc)

    def clear_population(self):
        """Clears cone populations in interior and exterior curves

        Arguments:
            self (circuit) : the object itself
        """
        self.interior_curve.true_cones = []
        self.exterior_curve.true_cones = []

    def clear_false_cones(self):
        """Clears false cone population

        Arguments:
            self (circuit) : the object itself
        """
        self.false_cones = []

    def plot(self):
        """Plots the circuit layout

        Arguments:
            self (circuit) : the object itself
        """
        curve_options = {
            "middle": self.middle_curve,
            "interior": self.interior_curve,
            "exterior": self.exterior_curve,
        }
        for curve_id in curve_options:
            if curve_options[curve_id] is not None:
                curve_options[curve_id].plot()
        for c in self.false_cones:
            plt.plot(c[0], c[1], "x", color="r")

    def to_image(
        self,
        saveimg: bool = False,
        saving_path: str = "",
        circuit_filename: str = "circuit.png",
        track_filename: str = "track.png",
        cones_filename: str = "cones.png",
        precision: int = 1000,
        showcones: bool = True,
    ):
        """Obtains an image representation of the circuit layout

        Arguments:
            self (circuit) : the object itself
            saveimg (bool) : whether to save the image or not
            precision (int) : the number of points to sample from the curves
            showcones (bool) : whether to show the cones or not
        Returns:
            image (np.array) : the image of the circuit layout
            model_label (np.array) : the image label of the circuit track
            model_input (np.array) : the image input of the circuit track
        """
        curve_options = {
            "middle": self.middle_curve,
            "interior": self.interior_curve,
            "exterior": self.exterior_curve,
        }
        curve_colorpalette = {
            "middle": image_utils.rgb_to_rgba(0, 255, 255),
            "interior": image_utils.rgb_to_rgba(255, 0, 0),
            "exterior": image_utils.rgb_to_rgba(0, 255, 0),
        }
        zones_colorpalette = {
            "inside": image_utils.rgb_to_rgba(255, 255, 0),
            "track": image_utils.rgb_to_rgba(0, 0, 255),
        }
        label_curve_colorpalette = {
            "middle": image_utils.rgb_to_rgba(0, 0, 0),
            "interior": image_utils.rgb_to_rgba(255, 255, 255),
            "exterior": image_utils.rgb_to_rgba(255, 255, 255),
        }
        label_zones_colorpalette = {
            "inside": image_utils.rgb_to_rgba(255, 255, 255),
            "track": image_utils.rgb_to_rgba(0, 0, 0),
        }
        objects_colorpalette = {"cone": image_utils.rgb_to_rgba(0, 0, 0)}
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        ax.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        gammadict = {}
        for curve_id in curve_options:
            if curve_options[curve_id] is not None:
                tt = np.linspace(
                    curve_options[curve_id].t[0],
                    curve_options[curve_id].t[-1],
                    precision,
                )
                gamma = curve_options[curve_id](tt)
                gammadict[curve_id] = gamma
                ax.plot(gamma[:, 0], gamma[:, 1], color=curve_colorpalette[curve_id])
                ax2.plot(
                    gamma[:, 0], gamma[:, 1], color=label_curve_colorpalette[curve_id]
                )
        interior_poly = Polygon(
            np.column_stack((gammadict["interior"][:, 0], gammadict["interior"][:, 1])),
            facecolor=zones_colorpalette["inside"],
            alpha=1,
            edgecolor="none",
        )
        exterior_poly = Polygon(
            np.column_stack((gammadict["exterior"][:, 0], gammadict["exterior"][:, 1])),
            facecolor=zones_colorpalette["track"],
            alpha=1,
            edgecolor="none",
        )
        ax.add_patch(exterior_poly)
        ax.add_patch(interior_poly)
        for curve_id in curve_options:
            if curve_options[curve_id] is not None:
                if len(curve_options[curve_id].true_cones) > 0 and showcones:
                    if len(curve_options[curve_id].true_cones_contaminated) == 0:
                        hc = np.array(
                            [
                                curve_options[curve_id](c)
                                for c in curve_options[curve_id].true_cones
                            ]
                        )
                        ax.scatter(hc[:, 0], hc[:, 1], c="black", s=10)
                        ax3.scatter(
                            hc[:, 0], hc[:, 1], color=objects_colorpalette["cone"], s=1
                        )
                    else:
                        hc = np.array(
                            [
                                curve_options[curve_id](c)
                                for c in curve_options[curve_id].true_cones_contaminated
                            ]
                        )
                        ax.scatter(hc[:, 0], hc[:, 1], c="black", s=10)
                        ax3.scatter(
                            hc[:, 0], hc[:, 1], color=objects_colorpalette["cone"], s=1
                        )
                        hc2 = np.array(self.false_cones)
                        try:
                            ax3.scatter(
                                hc2[:, 0],
                                hc2[:, 1],
                                color=objects_colorpalette["cone"],
                                s=1,
                            )
                            ax2.scatter(hc2[:, 0], hc2[:, 1], color="w", s=1)
                        except:
                            pass
        label_interior_poly = Polygon(
            np.column_stack((gammadict["interior"][:, 0], gammadict["interior"][:, 1])),
            facecolor=label_zones_colorpalette["inside"],
            alpha=1,
            edgecolor="none",
        )
        label_exterior_poly = Polygon(
            np.column_stack((gammadict["exterior"][:, 0], gammadict["exterior"][:, 1])),
            facecolor=label_zones_colorpalette["track"],
            alpha=1,
            edgecolor="none",
        )
        ax2.add_patch(label_exterior_poly)
        ax2.add_patch(label_interior_poly)
        im = image_utils.fig2data(fig)
        model_label = image_utils.fig2data(fig2)
        model_input = image_utils.fig2data(fig3)
        if saveimg:
            imageio.imsave(saving_path + circuit_filename, im)
            imageio.imsave(saving_path + track_filename, model_label)
            imageio.imsave(saving_path + cones_filename, model_input)
        model_input = np.where(model_input > 128, 255, 0)
        model_label = np.where(model_label > 128, 1.0, 0)
        return np.uint8(im), np.uint8(model_label), np.uint8(model_input)

    def to_csv(
        self, filename: str = None, false_cones: bool = True, track_points: bool = True
    ):
        """Saves a representation of the circuit in a csv file.

        Arguments:
            filename (str): The name of the file to be saved. If None, the file will be named
            "TILKE_generation_<date>_<time>".
                Default: None
            false_cones (bool): Whether to include the false cones in the csv file.
                Default: True
            track_points (bool): Whether to include the track points in the csv file.
                Default: True
        """
        if filename is None:
            filename = (
                f"TILKE_generation_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S-%f')}"
            )
        with open(f"{filename}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["EC-cones"])
            for t in self.exterior_curve.true_cones:
                writer.writerow(
                    [
                        round(self.exterior_curve[t][0], 4),
                        round(self.exterior_curve[t][1], 4),
                    ]
                )
            writer.writerow(["IC-cones"])
            for t in self.interior_curve.true_cones:
                writer.writerow(
                    [
                        round(self.interior_curve[t][0], 4),
                        round(self.interior_curve[t][1], 4),
                    ]
                )
            if false_cones and len(self.false_cones) > 0:
                writer.writerow(["F-cones"])
                for cone in self.false_cones:
                    writer.writerow([round(cone[0], 4), round(cone[1], 4)])
            if track_points:
                track_point_titles = ["EC-points", "IC-points", "MC-points"]
                track_lists = [
                    self.exterior_curve.data,
                    self.interior_curve.data,
                    self.middle_curve.data,
                ]
                track_point_lists = [
                    [[curve[0][i], curve[1][i]] for i in range(len(curve[0]))]
                    for curve in track_lists
                ]
                for i in range(len(track_point_titles)):
                    writer.writerow([track_point_titles[i]])
                    for point in track_point_lists[i]:
                        writer.writerow([round(point[0], 4), round(point[1], 4)])
