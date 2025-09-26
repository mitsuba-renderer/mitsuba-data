from __future__ import annotations

import numpy as np

import drjit as dr
import mitsuba as mi

def update_plugin(plugin_params, is_sun, t, eta):
    sp_sun, cp_sun = 0, 1 # dr.sincos(0.)
    st, ct = dr.sincos(dr.pi/2 - eta)

    if is_sun:
        plugin_params['sun_scale'] = 1.0
        plugin_params['sky_scale'] = 0.0
    else:
        plugin_params['sun_scale'] = 0.0
        plugin_params['sky_scale'] = 1.0

    plugin_params['turbidity'] = t
    plugin_params['sun_direction'] = mi.Vector3f(cp_sun * st, sp_sun * st, ct)
    plugin_params.update()

@dr.freeze
def get_rays(quad_points, weights, cos_cutoff):
    j = 0.5 * dr.pi * (1. - cos_cutoff)
    phi = dr.pi * (quad_points + 1)
    cos_theta = 0.5 * ((1. - cos_cutoff) * quad_points + (1 + cos_cutoff))

    phi, cos_theta = dr.meshgrid(phi, cos_theta)
    w_phi, w_cos_theta = dr.meshgrid(weights, weights)
    sin_phi, cos_phi = dr.sincos(phi)
    sin_theta = dr.safe_sqrt(1 - cos_theta * cos_theta)

    wo = mi.Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)

    return j, wo, w_phi * w_cos_theta

@dr.freeze
def evaluate_radiance(emitter, si):
    wavelengths = [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720]

    wav_res = dr.zeros(mi.ArrayXf, (11, 1))
    for i, wav in enumerate(wavelengths):
        si.wavelengths = wav
        radiance = emitter.eval(si)
        wav_res[i] = radiance[0]

    return wav_res


@dr.freeze
def sky_integrand(emitter, quad_points, quad_weights):
    j, sky_wo, weights = get_rays(quad_points, quad_weights, mi.Float(0))

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -sky_wo

    return j * evaluate_radiance(emitter, si) * weights


@dr.freeze
def sun_integrand(emitter, quad_points, quad_weights, sun_direction, sun_cos_cutoff):
    j, sun_wo, weights = get_rays(quad_points, quad_weights, sun_cos_cutoff)

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -mi.Frame3f(sun_direction).to_world(sun_wo)

    return j * evaluate_radiance(emitter, si) * weights


if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")

    sun_aperture = 0.5358
    sun_cos_cutoff = dr.cos(dr.deg2rad(0.5 * sun_aperture))

    # Load the sunsky emitter
    emitter = mi.load_dict({
        "type": "sunsky",
        "complex_sun": True,
        "albedo": 0.5,
        "sun_aperture": sun_aperture,
        "sun_direction": [0, 0, 1],
        "sun_scale": 0.0,
    })
    emitter_params = mi.traverse(emitter)

    res = 200
    quad_points, weights = mi.quad.gauss_legendre(res)

    res = (10, 30)
    turbs = np.linspace(1, 10, res[0])
    etas  = (np.arange(res[1]) * 3 + 2) / 180 * np.pi

    sky_spec_irradiance = np.zeros((*res, 11), dtype=np.float32)
    for i, turb in enumerate(turbs):
        for j, eta in enumerate(etas):
            update_plugin(emitter_params, False, turb, eta)

            integrand = sky_integrand(emitter, quad_points, weights)
            integrand = dr.sum(integrand, axis=1).numpy().squeeze()
            sky_spec_irradiance[i, j] = integrand

    sun_spec_irradiance = np.zeros((*res, 11), dtype=np.float32)
    for i, turb in enumerate(turbs):
        for j, eta in enumerate(etas):
            update_plugin(emitter_params, True, turb, eta)

            sun_dir = emitter_params['sun_direction']

            integrand = sun_integrand(emitter, quad_points, weights, sun_dir, sun_cos_cutoff)
            sun_spec_irradiance[i, j] = dr.sum(integrand, axis=1).numpy().squeeze()


    mi.tensor_io.write("output/sampling_data.bin",
        sky_irradiance=sky_spec_irradiance,
        sun_irradiance=sun_spec_irradiance
    )