import argparse
import glob
import os
import sys

import mitsuba

mitsuba.set_variant("scalar_spectral")

from mitsuba.core.xml import load_file
from mitsuba.core import Thread, Bitmap, Struct


def load_scene(filename, *args, **kwargs):
    """Prepares the file resolver and loads a Mitsuba scene from the given path."""
    fr = Thread.thread().file_resolver()
    here = os.path.dirname(__file__)
    fr.append(here)
    fr.append(os.path.join(here, filename))
    fr.append(os.path.dirname(filename))

    scene = load_file(filename, *args, **kwargs)
    assert scene is not None
    return scene


def render(scene, write_to):
    success = scene.integrator().render(scene, scene.sensors()[0])
    assert success
    film = scene.sensors()[0].film()
    film.bitmap().convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to)


def main(args):
    parser = argparse.ArgumentParser(prog='RenderDocImages')
    parser.add_argument('--force', action='store_true',
                        help='Force rerendering of all documentation images')
    parser.add_argument('--spp', default=1, type=int,
                        help='Samples per pixel')
    args = parser.parse_args()

    spp = args.spp
    force = args.force
    images_folder = os.path.join(os.path.dirname(__file__), '../images/render')
    os.makedirs(images_folder, exist_ok=True)
    scenes = glob.glob(os.path.join(os.path.dirname(__file__), '*.xml'))
    for scene_name in scenes:
        img_path = os.path.join(images_folder, os.path.split(scene_name)[-1].replace('.xml', '.jpg'))
        if not os.path.isfile(img_path) or force:
            scene_path = os.path.abspath(scene_name)
            scene = load_scene(scene_path, parameters=[('spp', str(spp))])
            render(scene, img_path)


if __name__ == "__main__":
    main(sys.argv[1:])
