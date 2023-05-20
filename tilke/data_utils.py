from tqdm import tqdm
from datetime import datetime
from tilke.Circuit import Circuit
import os


def get_sample_csv(
    n: int,
    path: str,
    contaminated: bool = False,
    false_cones: bool = True,
    track_points: bool = True,
):
    """Generates n csv files and saves them in the path directory.

    Arguments:
        n (int): The number of csv files to be generated.
        path (str): The path to the directory where the csv files will be saved.
        false_cones (bool): Whether to include the false cones in the csv file.
            Default: True
        track_points (bool): Whether to include the track points in the csv file.
            Default: True
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(range(n)):
        a = Circuit()
        a.populate_circuit()
        if contaminated:
            a.contaminate_cone_population()
        a.to_csv(
            f"{path}/TILKE_generation_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S-%f')}",
            false_cones,
            track_points,
        )
