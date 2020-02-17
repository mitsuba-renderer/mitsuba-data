import argparse
import glob
import os
import sys

import numpy as np
import mitsuba

# For some images, render in specific mode (e.g. polarized)
mode_override = {
    "bsdf_polarizer_aligned":   "scalar_spectral_polarized",
    "bsdf_polarizer_absorbing": "scalar_spectral_polarized",
    "bsdf_polarizer_middle":    "scalar_spectral_polarized",
    "integrator_stokes_cbox":   "scalar_mono_polarized",
}


def load_scene(filename, *args, **kwargs):
    """Prepares the file resolver and loads a Mitsuba scene from the given path."""
    from mitsuba.core.xml import load_file
    from mitsuba.core import Thread

    fr = Thread.thread().file_resolver()
    here = os.path.dirname(__file__)
    fr.append(here)
    fr.append(os.path.join(here, filename))
    fr.append(os.path.dirname(filename))

    scene = load_file(filename, *args, **kwargs)
    assert scene is not None
    return scene


def render(scene, write_to):
    from mitsuba.core import Bitmap, Struct

    success = scene.integrator().render(scene, scene.sensors()[0])
    assert success
    film = scene.sensors()[0].film()
    bitmap = film.bitmap(raw=False)
    if bitmap.channel_count() == 4:
        bitmap.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to)
    elif bitmap.channel_count() == 16:
        # Stokes output, rather specialized for 'integrator_stokes_cbox' scene atm.
        data_np = np.array(bitmap, copy=False).astype(np.float)
        s0 = data_np[:, :, 4]
        z = np.zeros(s0.shape)
        s1 = np.dstack([np.maximum(0, -data_np[:, :, 7]),  np.maximum(0, data_np[:, :, 7]),  z])
        s2 = np.dstack([np.maximum(0, -data_np[:, :, 10]), np.maximum(0, data_np[:, :, 10]), z])
        s3 = np.dstack([np.maximum(0, -data_np[:, :, 13]), np.maximum(0, data_np[:, :, 13]), z])
        Bitmap(s0).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to)
        Bitmap(s1).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to.replace('.jpg', '_s1.jpg'))
        Bitmap(s3).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to.replace('.jpg', '_s3.jpg'))
        Bitmap(s2).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to.replace('.jpg', '_s2.jpg'))
    else:
        if bitmap.channel_count() > 7:
            print('Unsupported number of AOV channels!')
            return

        data_np = np.array(bitmap, copy=False)[:, :, 4:]

        # normalize depth map
        if bitmap.channel_count() == 5:
            min_val = np.min(data_np)
            max_val = np.max(data_np)

            data_np = (data_np - min_val) / (max_val - min_val)

            bitmap = Bitmap(data_np, Bitmap.PixelFormat.Y)
        else:
            bitmap = Bitmap(data_np, Bitmap.PixelFormat.RGB)

        bitmap.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to)

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
    for scene_path in scenes:
        scene_name = os.path.split(scene_path)[-1][:-4]
        if scene_name in mode_override.keys():
            mitsuba.set_variant(mode_override[scene_name])
        else:
            mitsuba.set_variant("scalar_spectral")

        img_path = os.path.join(images_folder, scene_name + ".jpg")
        if not os.path.isfile(img_path) or force:
            scene_path = os.path.abspath(scene_path)
            scene = load_scene(scene_path, parameters=[('spp', str(spp))])
            render(scene, img_path)


if __name__ == "__main__":
    main(sys.argv[1:])
