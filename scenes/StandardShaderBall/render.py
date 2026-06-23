import mitsuba as mi

# Renders the scene spectrally (set variant to 'scalar_rgb' for RGB rendering)
mi.set_variant('scalar_spectral')

# Loads the scene into memory
scene = mi.load_file("StandardShaderBall.xml")

# Renders the scene. Render variables can be set in StandardShaderBall.xml
image = mi.render(scene)

# Writes the image to disk
mi.util.write_bitmap("render.png", image)