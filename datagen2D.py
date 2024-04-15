import logging
import math
import os

import fdtd
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import timeit

#from utils import Timer



from pde import PDEConfig, Maxwell3D

logger = logging.getLogger(__name__)

class Timer:
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start
        
def generate_trajectories_maxwell(
    pde: PDEConfig,
    mode: str,
    num_samples: int,
    dirname: str = "data",
    n_parallel: int = 1,
    seed: int = 42,
) -> None:
    """
    Generate data trajectories for 3D Maxwell equations
    Args:
        pde (PDEConfig): pde at hand [Maxwell3D]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create

    Returns:
        None
    """

    fdtd.set_backend("numpy")
    pde_string = str(pde)
    logger.info(f"Equation: {pde_string}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {num_samples}")

    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed)]))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    dataset = h5f.create_group(mode)

    d_field, h_field = {}, {}

    print(pde)
    nt, nx, ny, nz = pde.grid_size[0], pde.grid_size[1], pde.grid_size[2], pde.grid_size[3]

    d_field = dataset.create_dataset(
        "d_field",
        (num_samples, nt, nx, ny, 1, 3),
        dtype=float,
    )
    h_field = dataset.create_dataset(
        "h_field",
        (num_samples, nt, nx, ny, 1, 3),
        dtype=float,
    )

    def genfunc(idx, s):
        rng = np.random.RandomState(idx + s)
        # Initialize grid and light sources
        grid = fdtd.Grid(
            (pde.L, pde.L, pde.L),
            grid_spacing=pde.grid_spacing,
            permittivity=pde.permittivity,
            permeability=pde.permeability,
        )

        grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
        grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
        grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

        outer_area = (pde.n_large - pde.n) // 2


        grid[2:12, 40:47, 0] = fdtd.Object(permittivity=1.7**2, name="object")


       
        for i in range(6):
            lengthx = rng.randint(2, 6)
            startx = rng.randint(0, outer_area - lengthx)
            lengthy = rng.randint(2, 6)
            starty = rng.randint(0, 16 - lengthy)
            pointz = rng.randint(0, 16)
            ampl = rng.rand() * pde.amplitude
            ps = rng.uniform(low=0.0, high=2 * math.pi)
            p = rng.randint(0, 2)
            polar = ["x", "y"]
            period = pde.wavelength / pde.sol * rng.uniform(low=0.001, high=1e3)
            grid[startx : startx + lengthx, starty : starty + lengthy, 0] = fdtd.PlaneSource(
                period=period,
                amplitude=ampl,
                name=f"planesourcexy{i}",
                phase_shift=ps,
                polarization=polar[p],
            )

        d_field_, h_field_ = [], []
        grid.run(pde.skip_nt, progress_bar=False)
        for i in range(0, pde.nt):
            grid.run(pde.sample_rate, progress_bar=False)
            d_field_.append(grid.E[outer_area:-outer_area, outer_area:-outer_area, 0, :].copy())
            h_field_.append(grid.H[outer_area:-outer_area, outer_area:-outer_area, 0, :].copy())

        return np.array(d_field_), np.array(h_field_)

    with Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        d_field_ls, h_field_ls = zip(
            *Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples)))
        )

    logger.info(f"Took {gentime.dt:.3f} seconds")
    del rngs
    import gc

    gc.collect()

    with Timer() as writetime:
        for idx in range(num_samples):
            # Saving the trajectories
            d_field[idx : (idx + 1), ...] = np.expand_dims(d_field_ls[idx], axis = 3)
            h_field[idx : (idx + 1), ...] = np.expand_dims(h_field_ls[idx], axis = 3)

    logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")
    print()
    print("Data saved")
    print()
    print()
    h5f.close()

max = Maxwell3D(nt = 12, n = 48, n_large = 96, sample_rate = 25)
max2 = Maxwell3D(nt = 12,n = 48, n_large = 96,sample_rate = 25)
max3 = Maxwell3D(nt = 12, n = 48, n_large = 96,sample_rate = 25)


'''
for i in range(500 // 32):
    generate_trajectories_maxwell(
        pde = max,
        mode = "train",
        num_samples = 32,
        dirname  = "drive/MyDrive/maxwell/data_train2D_obst",
        n_parallel = 1,
        seed = i+30000)

for i in range(100 // 32):
    generate_trajectories_maxwell(
        pde = max2,
        mode = "val",
        num_samples = 32,
        dirname  = "drive/MyDrive/maxwell/data_val2D_obst",
        n_parallel = 1,
        seed = i+43000)
'''
for i in range(100 // 32):
    generate_trajectories_maxwell(
        pde = max3,
        mode = "test",
        num_samples = 32,
        dirname  = "drive/MyDrive/maxwell/data_test2D_obst_unseen",
        n_parallel = 1,
        seed = i+30)

