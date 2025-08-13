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
import struct
import sys

def size_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def write_tensor(filename, align=8, **kwargs):
    with open(filename, 'wb') as f:
        # Identifier
        f.write('tensor_file\0'.encode('utf8'))

        # Version number
        f.write(struct.pack('<BB', 1, 0))

        # Number of fields
        f.write(struct.pack('<I', len(kwargs)))

        # Maps to Struct.EType field in Mitsuba
        dtype_map = {
            np.uint8: 1,
            np.int8: 2,
            np.uint16: 3,
            np.int16: 4,
            np.uint32: 5,
            np.int32: 6,
            np.uint64: 7,
            np.int64: 8,
            np.float16: 9,
            np.float32: 10,
            np.float64: 11
        }

        offsets = {}
        fields = dict(kwargs)

        # Write all fields
        for k, v in fields.items():
            if type(v) is str:
                v = np.frombuffer(v.encode('utf8'), dtype=np.uint8)
            else:
                v = np.ascontiguousarray(v)
            fields[k] = v

            # Field identifier
            label = k.encode('utf8')
            f.write(struct.pack('<H', len(label)))
            f.write(label)

            # Field dimension
            f.write(struct.pack('<H', v.ndim))

            found = False
            for dt in dtype_map.keys():
                if dt == v.dtype:
                    found = True
                    f.write(struct.pack('B', dtype_map[dt]))
                    break
            if not found:
                raise Exception("Unsupported dtype: %s" % str(v.dtype))

            # Field offset (unknown for now)
            offsets[k] = f.tell()
            f.write(struct.pack('<Q', 0))

            # Field sizes
            f.write(struct.pack('<' + ('Q' * v.ndim), *v.shape))

        for k, v in fields.items():
            # Set field offset
            pos = f.tell()

            # Pad to requested alignment
            pos = (pos + align - 1) // align * align

            f.seek(offsets[k])
            f.write(struct.pack('<Q', pos))
            f.seek(pos)

            # Field data
            v.tofile(f)

        print('Wrote \"%s\" (%s)' % (filename, size_fmt(f.tell())))


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
    arr = arr.astype(np.float32)
    # Sort the data by turbidity and elevation
    sort_args = np.lexsort([arr[::, 1], arr[::, 0]])
    simplified_arr = arr[sort_args, 2:]

    # Convert the elevation to zenith angle on mu_theta
    simplified_arr[::, 1] = np.pi/2 - simplified_arr[::, 1]
    simplified_arr = simplified_arr.reshape((9, 30, 5, 5))
    write_tensor("../output/tgmm_tables.bin", tgmm_tables=simplified_arr)
