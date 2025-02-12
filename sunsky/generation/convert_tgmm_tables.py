#!/usr/bin/env python
"""
Usage: extract_tgmm_tables.py {path_to_model}

This script extracts and converts the necessary data from a CSV containing the
TGMM coefficients of the Hosek[1] sky model to a binary file which is later used
by the `sunsky` plugin.

The original CSV dataset can be found at:
https://github.com/cgaueb/tgmm_sky_sampling [2]

The binary file is written to `../output/tgmm_tables.bin` in the current folder.

[1] Lukáš Hošek, Alexander Wilkie, 2012.
    An analytic model for full spectral sky-dome radiance.
    ACM Trans. Graph. 31, 4, Article 95 (July 2012), 9 pages.
    https://doi.org/10.1145/2185520.2185591

[2] Nick Vitsas, Konstantinos Vardis, and Georgios Papaioannou. 2021.
    Sampling Clear Sky Models using Truncated Gaussian Mixtures.
    Eurographics Symposium on Rendering 
    https://doi.org/10.2312/sr.20211288
"""

import numpy as np
import pandas as pd
import mitsuba as mi
import sys


def write_array(output_filename, array, shape):
    fstream = mi.FileStream(output_filename, mi.FileStream.EMode.ETruncReadWrite)

    # =============== Write headers ===============
    fstream.write_uint8(ord('S'))
    fstream.write_uint8(ord('K'))
    fstream.write_uint8(ord('Y'))
    fstream.write_uint32(0)

    # =============== Write dimensions ===============
    fstream.write_uint64(len(shape))

    for dim_length in shape:
        fstream.write_uint64(dim_length);

        if dim_length == 0:
            raise RuntimeError("Got dimension with 0 elements!")

    # ==================== Write data ====================
    for value in array:
        fstream.write_float(value)

    fstream.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Exactly one arugment is expected: the path to CSV "
                           "data")
    filename = sys.argv[1]

    # Delete unused data
    df = pd.read_csv(filename)
    df.pop('RMSE')
    df.pop('MAE')
    df.pop('Volume')
    df.pop('Normalization')
    df.pop('Azimuth')

    arr = df.to_numpy()

    # Sort the data by turbidity and elevation
    sort_args = np.lexsort([arr[::, 1], arr[::, 0]])
    simplified_arr = arr[sort_args, 2:]

    # Convert the elevation to zenith angle on mu_theta
    simplified_arr[::, 1] = np.pi/2 - simplified_arr[::, 1]

    shape = (9, 30, 5, 5)
    write_array("../output/tgmm_tables.bin", np.ravel(simplified_arr), shape)
