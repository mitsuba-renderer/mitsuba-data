/*
   This executable generates binary files that are meant to be used by the
   `sunsky` plugin.

   It users headers from the authors' work [1][2]. They can be found at:
   https://cgg.mff.cuni.cz/projects/SkylightModelling/

   Most of the constants defined in this executable must exactly match those
   of the `sunsky` plugin for them to be compatible.

[1] Lukáš Hošek, Alexander Wilkie, 2012.
    An analytic model for full spectral sky-dome radiance.
    ACM Trans. Graph. 31, 4, Article 95 (July 2012), 9 pages.
    https://doi.org/10.1145/2185520.2185591

[2] Lukáš Hošek, Alexander Wilkie. 2013.
    Adding a Solar-Radiance Function to the Hošek-Wilkie Skylight Model.
    IEEE Computer Graphics and Applications 33, 3 (2013), 44–52.
    https://doi.org/10.1109/MCG.2013.18
*/

#include "ArHosekSkyModelData_Spectral.h"
#include "ArHosekSkyModelData_RGB.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>
#include <unordered_map>

/// Number of spectral channels in the skylight model
static const constexpr size_t WAVELENGTH_COUNT = 11;
/// Number of turbidity levels in the skylight model
static const constexpr size_t TURBITDITY_LVLS = 10;
/// Number of albedo levels in the skylight model
static const constexpr size_t ALBEDO_LVLS = 2;

/// Wavelengths used in the skylight model
template <typename Float>
static constexpr Float WAVELENGTHS[WAVELENGTH_COUNT] = {
    320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720
};

/// Number of control points for interpolation in the skylight model
static const constexpr size_t SKY_CTRL_PTS = 6;
/// Number of parameters for the skylight model equation
static const constexpr size_t SKY_PARAMS = 9;

/// Number of control points for interpolation in the sun model
static const constexpr size_t SUN_CTRL_PTS = 4;
/// Number of segments for the piecewise polynomial in the sun model
static const constexpr size_t SUN_SEGMENTS = 45;
/// Number of coefficients for the sun's limb darkening
static const constexpr size_t SUN_LD_PARAMS = 6;

#define CIE_SAMPLES 95

#define F_DIM 5
#define L_DIM 4

enum class SunDataShapeIdx: uint32_t { WAVELENGTH = 0, TURBIDITY, SUN_SEGMENTS, SUN_CTRL_PTS };
enum class SkyDataShapeIdx: uint32_t { WAVELENGTH = 0, ALBEDO, TURBIDITY, CTRL_PT, PARAMS };

// =========================== SHAPES OF THE ORIGINAL WILKIE-HOSEK DATASETS ============================
constexpr size_t solar_shape[4] = {WAVELENGTH_COUNT, TURBITDITY_LVLS, SUN_SEGMENTS, SUN_CTRL_PTS};
constexpr size_t limb_darkening_shape[2] = {WAVELENGTH_COUNT, 6};

constexpr size_t f_spec_shape[F_DIM] = {WAVELENGTH_COUNT, ALBEDO_LVLS, TURBITDITY_LVLS, SKY_CTRL_PTS, SKY_PARAMS};
constexpr size_t l_spec_shape[L_DIM] = {WAVELENGTH_COUNT, ALBEDO_LVLS, TURBITDITY_LVLS, SKY_CTRL_PTS};

constexpr size_t f_tri_shape[F_DIM] = {3, ALBEDO_LVLS, TURBITDITY_LVLS, SKY_CTRL_PTS, SKY_PARAMS};
constexpr size_t l_tri_shape[L_DIM] = {3, ALBEDO_LVLS, TURBITDITY_LVLS, SKY_CTRL_PTS};

/// Type of a field in the \c Struct
enum class Type : uint32_t {
    // Invalid/unspecified
    Invalid = 0,

    // Signed and unsigned integer values
    UInt8,  Int8,
    UInt16, Int16,
    UInt32, Int32,
    UInt64, Int64,

    // Floating point values
    Float16, Float32, Float64,
};


/// Helper struct to link the dataset metadata
struct TensorField {
	Type dtype;

	std::vector<size_t> shape;

	const void* data;

	size_t offset = -1;
};

const TensorField f_spectral = {
	Type::Float64,
	std::vector<size_t>(f_spec_shape, f_spec_shape + F_DIM),
    datasets
};

const TensorField l_spectral = {
	Type::Float64,
	std::vector<size_t>(l_spec_shape, l_spec_shape + L_DIM),
    datasetsRad
};

const TensorField f_RGB = {
	Type::Float64,
	std::vector<size_t>(f_tri_shape, f_tri_shape + F_DIM),
    datasetsRGB
};

const TensorField l_RGB = {
	Type::Float64,
	std::vector<size_t>(l_tri_shape, l_tri_shape + L_DIM),
    datasetsRGBRad
};

const TensorField solar_dataset = {
	Type::Float64,
	std::vector<size_t>(solar_shape, solar_shape + 4),
    solarDatasets
};

const TensorField limb_darkening_dataset = {
	Type::Float64,
	std::vector<size_t>(limb_darkening_shape, limb_darkening_shape + 2),
    limbDarkeningDatasets
};


uint64_t get_tensor_size(const std::vector<size_t>& shape) {
	if (shape.size() == 0)
		throw std::runtime_error("Empty tensor shape");
	
	uint64_t tensor_size = shape[0];
	for (size_t i = 1; i < shape.size(); ++i) {
		if (shape[i] == 0)
			throw std::runtime_error("Got empty dimension");
		
		tensor_size *= shape[i];
	}
	
	return tensor_size;
}

/** 
 * \brief Tiny file stream
 */
class TensorFileStream {
public:
    TensorFileStream(const std::string &p) : m_file(new std::fstream) {
        m_file->open(p, std::ios::binary | std::ios::out |
                            std::ios::trunc);
        if (!m_file->good())
            throw std::runtime_error("I/O error while opening file!");

		// Header: 12 bytes
		write("tensor_file", 12);

		// Version number: 2 * 1 byte
		write<uint8_t>(1);
		write<uint8_t>(0);

		// Number of fields to be written on close
		write<uint32_t>(-1);
    }

    void write(const void *p, size_t size) {
        m_file->write((char *) p, size);

        if (!m_file->good()) {
            m_file->clear();
            throw std::runtime_error("I/O error while writing to file!");
        }
    }

	template <typename T> void write(const T &value) {
        write(&value, sizeof(T));
    }

    void emplace_field(const std::string_view name, TensorField& field) {
		m_fields[name] = field;

		// Write tensor name
        write<uint16_t>(name.length());
		write(name.data(), name.length());
		// Write number of dimensions
		write<uint16_t>(field.shape.size());

		write<uint8_t>((uint8_t) field.dtype);

		// Save offset entry for this field, will be written to later
		m_fields[name].offset = m_file->tellp();
		// Write dummy offset
		write<uint64_t>(-1);

		for (const size_t size_value: field.shape)
			write<uint64_t>(size_value);
		
    }

	~TensorFileStream() { close(); };

    void close() {
		// Write number of fields
		uint64_t metadata_end = m_file->tellp();
		m_file->seekp(nb_fields_offset);
		write<uint32_t>(m_fields.size());
		m_file->seekp(metadata_end);
		
		for (auto& [name, field] : m_fields) {
			uint64_t tensor_offset = m_file->tellp();
			m_file->seekp(field.offset);
			write<uint64_t>(tensor_offset);
			m_file->seekp(tensor_offset);

			uint64_t tensor_size = get_tensor_size(field.shape);

			// TODO may be needed to generalize
			write(field.data, tensor_size * (field.dtype == Type::Float64 ? 8 : 4));

			free((void *) field.data);

		}

		m_file->close(); 
	}

private:
	const size_t nb_fields_offset = 14;
    std::unique_ptr<std::fstream> m_file;
	std::unordered_map<std::string_view, TensorField> m_fields;
};


// CIE 1931 table
static const double cie1931_tbl[95 * 3] = {
    double(0.000129900000), double(0.000232100000), double(0.000414900000), double(0.000741600000),
    double(0.001368000000), double(0.002236000000), double(0.004243000000), double(0.007650000000),
    double(0.014310000000), double(0.023190000000), double(0.043510000000), double(0.077630000000),
    double(0.134380000000), double(0.214770000000), double(0.283900000000), double(0.328500000000),
    double(0.348280000000), double(0.348060000000), double(0.336200000000), double(0.318700000000),
    double(0.290800000000), double(0.251100000000), double(0.195360000000), double(0.142100000000),
    double(0.095640000000), double(0.057950010000), double(0.032010000000), double(0.014700000000),
    double(0.004900000000), double(0.002400000000), double(0.009300000000), double(0.029100000000),
    double(0.063270000000), double(0.109600000000), double(0.165500000000), double(0.225749900000),
    double(0.290400000000), double(0.359700000000), double(0.433449900000), double(0.512050100000),
    double(0.594500000000), double(0.678400000000), double(0.762100000000), double(0.842500000000),
    double(0.916300000000), double(0.978600000000), double(1.026300000000), double(1.056700000000),
    double(1.062200000000), double(1.045600000000), double(1.002600000000), double(0.938400000000),
    double(0.854449900000), double(0.751400000000), double(0.642400000000), double(0.541900000000),
    double(0.447900000000), double(0.360800000000), double(0.283500000000), double(0.218700000000),
    double(0.164900000000), double(0.121200000000), double(0.087400000000), double(0.063600000000),
    double(0.046770000000), double(0.032900000000), double(0.022700000000), double(0.015840000000),
    double(0.011359160000), double(0.008110916000), double(0.005790346000), double(0.004109457000),
    double(0.002899327000), double(0.002049190000), double(0.001439971000), double(0.000999949300),
    double(0.000690078600), double(0.000476021300), double(0.000332301100), double(0.000234826100),
    double(0.000166150500), double(0.000117413000), double(0.000083075270), double(0.000058706520),
    double(0.000041509940), double(0.000029353260), double(0.000020673830), double(0.000014559770),
    double(0.000010253980), double(0.000007221456), double(0.000005085868), double(0.000003581652),
    double(0.000002522525), double(0.000001776509), double(0.000001251141),

    double(0.000003917000), double(0.000006965000), double(0.000012390000), double(0.000022020000),
    double(0.000039000000), double(0.000064000000), double(0.000120000000), double(0.000217000000),
    double(0.000396000000), double(0.000640000000), double(0.001210000000), double(0.002180000000),
    double(0.004000000000), double(0.007300000000), double(0.011600000000), double(0.016840000000),
    double(0.023000000000), double(0.029800000000), double(0.038000000000), double(0.048000000000),
    double(0.060000000000), double(0.073900000000), double(0.090980000000), double(0.112600000000),
    double(0.139020000000), double(0.169300000000), double(0.208020000000), double(0.258600000000),
    double(0.323000000000), double(0.407300000000), double(0.503000000000), double(0.608200000000),
    double(0.710000000000), double(0.793200000000), double(0.862000000000), double(0.914850100000),
    double(0.954000000000), double(0.980300000000), double(0.994950100000), double(1.000000000000),
    double(0.995000000000), double(0.978600000000), double(0.952000000000), double(0.915400000000),
    double(0.870000000000), double(0.816300000000), double(0.757000000000), double(0.694900000000),
    double(0.631000000000), double(0.566800000000), double(0.503000000000), double(0.441200000000),
    double(0.381000000000), double(0.321000000000), double(0.265000000000), double(0.217000000000),
    double(0.175000000000), double(0.138200000000), double(0.107000000000), double(0.081600000000),
    double(0.061000000000), double(0.044580000000), double(0.032000000000), double(0.023200000000),
    double(0.017000000000), double(0.011920000000), double(0.008210000000), double(0.005723000000),
    double(0.004102000000), double(0.002929000000), double(0.002091000000), double(0.001484000000),
    double(0.001047000000), double(0.000740000000), double(0.000520000000), double(0.000361100000),
    double(0.000249200000), double(0.000171900000), double(0.000120000000), double(0.000084800000),
    double(0.000060000000), double(0.000042400000), double(0.000030000000), double(0.000021200000),
    double(0.000014990000), double(0.000010600000), double(0.000007465700), double(0.000005257800),
    double(0.000003702900), double(0.000002607800), double(0.000001836600), double(0.000001293400),
    double(0.000000910930), double(0.000000641530), double(0.000000451810),

    double(0.000606100000), double(0.001086000000), double(0.001946000000), double(0.003486000000),
    double(0.006450001000), double(0.010549990000), double(0.020050010000), double(0.036210000000),
    double(0.067850010000), double(0.110200000000), double(0.207400000000), double(0.371300000000),
    double(0.645600000000), double(1.039050100000), double(1.385600000000), double(1.622960000000),
    double(1.747060000000), double(1.782600000000), double(1.772110000000), double(1.744100000000),
    double(1.669200000000), double(1.528100000000), double(1.287640000000), double(1.041900000000),
    double(0.812950100000), double(0.616200000000), double(0.465180000000), double(0.353300000000),
    double(0.272000000000), double(0.212300000000), double(0.158200000000), double(0.111700000000),
    double(0.078249990000), double(0.057250010000), double(0.042160000000), double(0.029840000000),
    double(0.020300000000), double(0.013400000000), double(0.008749999000), double(0.005749999000),
    double(0.003900000000), double(0.002749999000), double(0.002100000000), double(0.001800000000),
    double(0.001650001000), double(0.001400000000), double(0.001100000000), double(0.001000000000),
    double(0.000800000000), double(0.000600000000), double(0.000340000000), double(0.000240000000),
    double(0.000190000000), double(0.000100000000), double(0.000049999990), double(0.000030000000),
    double(0.000020000000), double(0.000010000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000), double(0.000000000000),
    double(0.000000000000), double(0.000000000000), double(0.000000000000)
};

std::array<double, 3> linear_rgb_rec(double wavelength) {
    double CIE_MIN = 360;
    double CIE_MAX = 830;

    double t =
        (wavelength - CIE_MIN) * ((CIE_SAMPLES - 1) / (CIE_MAX - CIE_MIN));

    bool active = wavelength >= CIE_MIN && wavelength <= CIE_MAX;
    if (!active)
        return {0, 0, 0};

    uint32_t i0 = std::max(std::min((uint32_t) CIE_SAMPLES - 2, (uint32_t) t), 0u);
    uint32_t i1 = i0 + 1;

    double x0 = cie1931_tbl[i0 + 95 * 0];
    double y0 = cie1931_tbl[i0 + 95 * 1];
    double z0 = cie1931_tbl[i0 + 95 * 2];

    double x1 = cie1931_tbl[i1 + 95 * 0];
    double y1 = cie1931_tbl[i1 + 95 * 1];
    double z1 = cie1931_tbl[i1 + 95 * 2];

    /// Convert XYZ tristimulus values to ITU-R Rec. BT.709 linear RGB
    double mat[3][3] = {
        { 3.240479f, -1.537150f, -0.498535f},
        {-0.969256f,  1.875991f,  0.041556f}, 
        { 0.055648f, -0.204043f,  1.057311f}
    };

    auto matmul = [&](size_t row, const std::array<double, 3>& vec) -> double {
        double out = 0;
        for (size_t i = 0; i < 3; i ++)
            out += mat[row][i] * vec[i];

        return out;
    };

    double r0 = matmul(0, {x0, y0, z0});
    double g0 = matmul(1, {x0, y0, z0});
    double b0 = matmul(2, {x0, y0, z0});

    double r1 = matmul(0, {x1, y1, z1});
    double g1 = matmul(1, {x1, y1, z1});
    double b1 = matmul(2, {x1, y1, z1});


    double w1 = t - double(i0);
    double w0 = 1. - w1;
    double r = w0 * r0 + (w1 * r1);
    double g = w0 * g0 + (w1 * g1);
    double b = w0 * b0 + (w1 * b1);

    return {r, g, b};
}

 /// Serializes the Limb Darkening dataset to a tensor field
 TensorField get_limb_darkening_data() {
    const auto [dtype, shape, data, _] = limb_darkening_dataset;
	double** p_dataset = (double**) data;

	double* dataset = (double*) calloc(WAVELENGTH_COUNT * SUN_LD_PARAMS, sizeof(double));
    for (size_t w = 0; w < WAVELENGTH_COUNT; ++w)
		memcpy(dataset + w * SUN_LD_PARAMS, p_dataset[w], SUN_LD_PARAMS * sizeof(double));

	return TensorField {
		dtype, shape, dataset
	};
}

/// Precomputes the sun RGB dataset from the spectral dataset and
/// the limb darkening dataset
TensorField get_sun_data_rgb() {
    const auto [sun_rad_dtype, solar_shape, data_solar, _1] = solar_dataset;
    const auto [ld_dtype, ld_shape, data_ld, _2] = limb_darkening_dataset;
	double** p_dataset_solar = (double**) data_solar;
	double** p_dataset_ld = (double**) data_ld;

	TensorField result = {
		sun_rad_dtype, { TURBITDITY_LVLS, SUN_SEGMENTS, 3, SUN_CTRL_PTS, SUN_LD_PARAMS }, nullptr
	};

	size_t dst_idx = 0;
    double* buffer = (double*)calloc(TURBITDITY_LVLS * SUN_SEGMENTS * 3 * SUN_CTRL_PTS * SUN_LD_PARAMS, sizeof(double));

    std::array<double, 3> rectifier_rgb;
    for (size_t turb = 0; turb < TURBITDITY_LVLS; ++turb) {
        for (size_t segment = 0; segment < SUN_SEGMENTS; ++segment) {
            for (size_t rgb_idx = 0; rgb_idx < 3; ++rgb_idx) {
                for (size_t ctrl_pt = 0; ctrl_pt < SUN_CTRL_PTS; ++ctrl_pt) {
                    // Weird indices since their dataset goes backwards on the last index
                    const size_t sun_idx = turb * (SUN_SEGMENTS * SUN_CTRL_PTS) +
                                                  (segment + 1) * SUN_CTRL_PTS - (ctrl_pt + 1);
                    for (size_t ld_param_idx = 0; ld_param_idx < SUN_LD_PARAMS; ++ld_param_idx) {
                        // Convert from spectral to RGB
                        for (size_t lambda = 0; lambda < WAVELENGTH_COUNT; ++lambda) {
                            rectifier_rgb = linear_rgb_rec(WAVELENGTHS<double>[lambda]);
                            double rectifier = rectifier_rgb[rgb_idx];
                            buffer[dst_idx] += rectifier * p_dataset_solar[lambda][sun_idx] *
                                                    p_dataset_ld[lambda][ld_param_idx];
                        }

                        buffer[dst_idx] /= WAVELENGTH_COUNT;
                        ++dst_idx;
                    }
                }
            }
        }
    }

	return TensorField {
		sun_rad_dtype,
		{ TURBITDITY_LVLS, SUN_SEGMENTS, 3, SUN_CTRL_PTS, SUN_LD_PARAMS }, 
		buffer
	};
}

/// Re-orders the sun spectral dataset from the original dataset
/// From [wavelengths x turbidity x sun_segments x sun_ctrl_pts]
/// to   [turbidity x sun_segments x wavelengths x sun_ctrl_pts]
TensorField get_sun_data_spectral() {
    const auto [dtype, shape, data, _] = solar_dataset;
	double** p_dataset = (double**) data;

	size_t idx = 0;
	double* buffer = (double*) calloc(TURBITDITY_LVLS * SUN_SEGMENTS * WAVELENGTH_COUNT * SUN_CTRL_PTS, sizeof(double));
    for (size_t turb = 0; turb < TURBITDITY_LVLS; ++turb) {
        for (size_t segment = 0; segment < SUN_SEGMENTS; ++segment) {
            for (size_t lambda = 0; lambda < WAVELENGTH_COUNT; ++lambda) {
                for (size_t ctrl_pt = 0; ctrl_pt < SUN_CTRL_PTS; ++ctrl_pt) {
                    // Weird indices since their dataset goes backwards on the last index
                    const size_t src_global_offset =
                        turb * (SUN_SEGMENTS * SUN_CTRL_PTS) +
                        (segment + 1) * SUN_CTRL_PTS -
                        (ctrl_pt + 1);

                    buffer[idx++] = p_dataset[lambda][src_global_offset];
                }
            }
        }
    }

	return TensorField {
		dtype,
		{ shape[(uint32_t)SunDataShapeIdx::TURBIDITY], 
			shape[(uint32_t)SunDataShapeIdx::SUN_SEGMENTS], 
			shape[(uint32_t)SunDataShapeIdx::WAVELENGTH], 
			shape[(uint32_t)SunDataShapeIdx::SUN_CTRL_PTS]
		},
		buffer
	};
}


/// Re-orders the sky dataset from the original dataset
/// From [wavelengths/channels x albedo x turbidity x sky_ctrl_pts (x sky_params)]
/// to   [turbidity x albedo x sky_ctrl_pts x wavelengths/channels (x sky_params)]
/// \param dataset
///     Dataset metadata to write
TensorField get_sky_data(const TensorField& dataset) {
    const auto [dtype, shape, data, _] = dataset;
	double** p_dataset = (double**) data;

	uint64_t tensor_size = get_tensor_size(shape);
	std::vector<size_t> reordered_shape = {
		shape[(uint32_t) SkyDataShapeIdx::TURBIDITY],
		shape[(uint32_t) SkyDataShapeIdx::ALBEDO],
		shape[(uint32_t) SkyDataShapeIdx::CTRL_PT],
		shape[(uint32_t) SkyDataShapeIdx::WAVELENGTH]
	};

	if (shape.size() == F_DIM)
		reordered_shape.push_back(shape[(uint32_t)SkyDataShapeIdx::PARAMS]);

    const size_t nb_params = shape.size() == F_DIM ? SKY_PARAMS : 1,
                 nb_colors = shape[(uint32_t)SkyDataShapeIdx::WAVELENGTH];

    size_t dst_idx = 0;
    double* buffer = (double*)calloc(tensor_size, sizeof(double));

    // Converts from (11 x 2 x 10 x 6 x ...) to (10 x 2 x 6 x 11 x ...)
    for (size_t t = 0; t < TURBITDITY_LVLS; ++t) {
        for (size_t a = 0; a < ALBEDO_LVLS; ++a) {
            for (size_t ctrl_idx = 0; ctrl_idx < SKY_CTRL_PTS; ++ctrl_idx) {
                for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {
                    for (size_t param_idx = 0; param_idx < nb_params; ++param_idx) {
                        size_t src_global_offset =
                            a * (TURBITDITY_LVLS * SKY_CTRL_PTS * nb_params) +
                            t * (SKY_CTRL_PTS * nb_params) +
                            ctrl_idx * nb_params +
                            param_idx;
                        buffer[dst_idx++] = p_dataset[color_idx][src_global_offset];
                    }
                }
            }
        }
    }

	return TensorField {
		dtype, reordered_shape, buffer
	};
}

/// Generates the datasets files from the original, this function should not
/// be called for each render job, only when the dataset files are lost.
void write_sun_sky_model_data(const std::string &path) {
    TensorField sky_spec_params = get_sky_data(f_spectral),
				sky_spec_rad = get_sky_data(l_spectral),
				sky_rgb_params = get_sky_data(f_RGB),
				sky_rgb_rad = get_sky_data(l_RGB),
				sun_spec_rad = get_sun_data_spectral(),
				sun_rgb_rad = get_sun_data_rgb(),
    			sun_ld_field = get_limb_darkening_data();

	TensorFileStream file(path + "/sunsky_datasets.bin");
	file.emplace_field("sky_params_spec", sky_spec_params);
	file.emplace_field("sky_rad_spec", sky_spec_rad);
	file.emplace_field("sky_params_rgb", sky_rgb_params);
	file.emplace_field("sky_rad_rgb", sky_rgb_rad);
	file.emplace_field("sun_rad_spec", sun_spec_rad);
	file.emplace_field("sun_ld_spec", sun_ld_field);
	file.emplace_field("sun_rad_rgb", sun_rgb_rad);

}

int main(int argc, char *argv[]) {
    std::string path = ".";
    if (argc == 2) {
        write_sun_sky_model_data(argv[1]);
    } else {
        std::cerr << "This executable expects exactly one "
                     "argument: a path to the output folder! Aborting..."
                  << std::endl;
    }

    return 0;
}
