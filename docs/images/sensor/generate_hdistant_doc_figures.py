import matplotlib.pyplot as plt
import numpy as np

import mitsuba

mitsuba.set_variant("scalar_rgb")

from mitsuba.core import ScalarTransform4f
from mitsuba.core.xml import load_dict

direction_r = [1, 0, -1]
direction_g = [1, 1, -1]
direction_b = [0, 1, -1]
film_resolution = 32


def scene_dict(sensor_to_world=None):
    if sensor_to_world is None:
        sensor_to_world = ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=[0, 0, 1],
            up=[0, 1, 0],
        )

    return {
        "type": "scene",
        "shape": {
            "type": "rectangle",
            "bsdf": {
                "type": "roughconductor"
            },
        },
        "illumination_r": {
            "type": "directional",
            "direction": direction_r,
            "irradiance": {
                "type": "rgb",
                "value": [1, 0, 0],
            },
        },
        "illumination_g": {
            "type": "directional",
            "direction": direction_g,
            "irradiance": {
                "type": "rgb",
                "value": [0, 1, 0],
            },
        },
        "illumination_b": {
            "type": "directional",
            "direction": direction_b,
            "irradiance": {
                "type": "rgb",
                "value": [0, 0, 1],
            },
        },
        "hdistant": {
            "type": "hdistant",
            "to_world": sensor_to_world,
            "sampler": {
                "type": "independent",
                "sample_count": 3200,
            },
            "film": {
                "type": "hdrfilm",
                "width": film_resolution,
                "height": film_resolution,
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": {
                    "type": "box"
                },
            }
        },
        #"camera": {
        #    "type": "perspective",
        #    "to_world": ScalarTransform4f.look_at(
        #        origin=[5, 5, 5],
        #        target=[0, 0, 0],
        #        up=[0, 0, 1],
        #    ),
        #    "sampler": {
        #        "type": "independent",
        #        "sample_count": 32,
        #    },
        #    "film": {
        #        "type": "hdrfilm",
        #        "width": 320,
        #        "height": 240,
        #        "pixel_format": "luminance",
        #        "component_format": "float32",
        #    }
        #},
        "integrator": {
            "type": "path"
        },
    }


for name, sensor_to_world in {
        "default":
        ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=[0, 0, 1],
            up=[0, 1, 0],
        ),
        "rotated":
        ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=[0, 0, 1],
            up=[1, 1, 0],
        ),
}.items():
    scene = load_dict(scene_dict(sensor_to_world=sensor_to_world))
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)

    # Plot recorded leaving radiance
    img = np.array(sensor.film().bitmap()).squeeze()
    img -= np.min(img)
    img = img / np.max(img)
    plt.imshow(img, origin="lower")

    # Add illumination setup
    from mitsuba.core.warp import uniform_hemisphere_to_square

    # -- We must convert emitter directions to the surface scattering frame
    def direction_to_pixel_coords(direction):
        d = -np.array(sensor_to_world.inverse().transform_vector(direction))
        d = d / np.linalg.norm(d)
        return uniform_hemisphere_to_square(d) * float(film_resolution)

    plt.scatter(*direction_to_pixel_coords(direction_r), color="r")
    plt.scatter(*direction_to_pixel_coords(direction_g), color="g")
    plt.scatter(*direction_to_pixel_coords(direction_b), color="b")

    # -- Add up and target directions to film view
    center = np.array([
        0.5 * float(film_resolution),
        0.5 * float(film_resolution),
    ])
    up = 0.75 * np.array([0.0, 0.5 * float(film_resolution)])
    orange = (1, 0.4, 0)
    plt.arrow(
        *center,
        *up,
        width=0.3,
        head_width=1,
        color=orange,
    )
    plt.scatter(*center, color=orange)
    plt.scatter(*center, color="none", s=250, edgecolors=orange)

    # Add axis labels
    plt.xlabel("pixel index")
    plt.ylabel("pixel index")

    plt.savefig(f"sensor_hdistant_{name}.svg")
    plt.close()
