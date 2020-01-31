import os
from os.path import realpath, dirname, join, isfile
import subprocess

CURRENT_DIRECTORY = realpath(dirname(__file__))
MTS1_DIST = realpath(join(CURRENT_DIRECTORY, '../../../../../mitsuba/dist'))
MTS2_DIST = realpath(join(CURRENT_DIRECTORY, '../../../../build/dist'))
assert isfile(join(MTS1_DIST, 'mitsuba'))

scene_variants = {
    'cbox_spectral': 'cbox-spectral.xml',
    'cbox_rgb': 'cbox-rgb.xml',
}
modes = [
    'scalar_rgb',
    'scalar_spectral',
    # 'packet_rgb',
    # 'packet_spectral',
]


def main():
    i = 0
    for variant, scene_fname in scene_variants.items():
        # Mitsuba 1 reference
        mts1_fname = join(CURRENT_DIRECTORY, 'mts1-' + scene_fname)
        out = join(CURRENT_DIRECTORY, '{:02d}-mts1-{}.exr'.format(i, variant))
        subprocess.check_call(['./mitsuba', mts1_fname, '-o', out], cwd=MTS1_DIST)
        i += 1

        # Mitsuba 2, different modes
        for mode in modes:
            mts2_fname = join(CURRENT_DIRECTORY, scene_fname)
            out = join(CURRENT_DIRECTORY, '{:02d}-mts2-{}--{}.exr'.format(i, variant, mode))
            subprocess.check_call(['./mitsuba', mts2_fname,
                                   '-m', mode,
                                   '-o', out], cwd=MTS2_DIST)
            i += 1


if __name__ == '__main__':
    main()
