# Ray tracing an Ellis Wormhole

![View through a traversable wormhole into another universe](media/main.png)

▶ **[Full render on YouTube](https://www.youtube.com/watch?v=IK4DtV514to)** 

A GPU accelerated ray tracing program showing what a camera would see near and while travelling through an Ellis-Bronnikov wormhole.

## Running the program

```bash
pip install -r ../requirements.txt
python ellis_wormhole.py
```

By default this renders one 360° frame from outside the wormhole as `render.png`. Adjustable settings are at the top of the file.

| Setting | Meaning |
|---|---|
| `throat_radius` | Radius of the wormhole's throat |
| `MODE` | `"image"` generates one frame, `"video"` flies the camera through the throat |
| `width, height` | Resolution (about 1.1 GB GPU memory at 2880×1440, plus the sky images) |
| `VR_360` | Toggles 360° frames (requires a 2:1 resolution, e.g. 3840×1920) |
| `CAMERA_L` | Camera position for a single image (negative = the star map's universe) |
| `l_start, l_end` | Start and end of the camera's path in video mode |

**Sky images (optional):** Each universe has its own sky. download the 8K EXR from NASA SVS *Deep Star Maps 2020* as `assets/starmap_8k.exr` (the l < 0 universe) and the alternate starmap from Second sky: ESO/S. Brunier, *The Milky Way panorama* Fullsize Original tif as `assets/eso_milky_way.tif` (the l > 0 universe). 
Without them, default white and orange checkerboards are used.


**Video:** video frames are saved to `frames/`; they can be stitched together into a single video with ffmpeg:

`ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4`

This code runs on NVIDIA GPUs via CUDA and falls back to Vulkan/Metal/CPU if unavailable (slower).


## The physics

The Ellis-Bronnikov wormhole is described by the metric:

$$ds^2 = -dt^2 + dl^2 + (l^2 + b_0^2)(d\theta^2 + \sin^2\theta\, d\phi^2)$$

where $l$ is the proper radial distance. Positive and negative $l$ correspond to two separate universes which are joined at the **throat** ($l = 0$), a sphere of radius $b_0$.

Unlike a black hole, the wormhole has **no event horizon** and **no gravitational redshift** so all the light bending comes from the curved shape of space itself. Light can also form unstable circular orbits at the throat, producing a bright lensed ring.

The equations of motion of null geodesics (light rays) were derived by hand from the Euler-Lagrange equations:

$$\ddot{l} = l\,(\dot\theta^2 + \sin^2\theta\,\dot\phi^2)$$

$$\ddot\theta = \sin\theta\cos\theta\,\dot\phi^2 - \frac{2l\,\dot l\,\dot\theta}{l^2 + b_0^2}, \qquad \ddot\phi = -2\cot\theta\,\dot\theta\,\dot\phi - \frac{2l\,\dot l\,\dot\phi}{l^2 + b_0^2}$$

Unfortunately, a real traversable wormhole would require exotic matter with negative energy density, which has not been observed in the real world. This simulation shows what one would look like if it could exist.

## How the code works

- **Camera:** It is placed directly in wormhole coordinates $(l, \theta, \phi)$. Each pixel's ray direction is converted from the camera's local frame to coordinate velocities (by dividing the angular parts by the local radius $\sqrt{l^2 + b_0^2}$).
- **Integration:** Each ray is integrated with the same adaptive RK45 integrator as the Kerr metric ray tracer until they escape to infinity in either universe.
- **Skybox:** the sign of $l$ decides which universe's sky the ray portrays.
- **Display:** The light from each photon is compiled together to form an image (with anti-aliasing). Reinhard tone mapping and gamma correction are then applied as visual effects (not physical). 
- **GPU:**  In the Taichi kernel, rays are processed in parallel by the GPU, independently from each other.


## Problems I solved

| Version | What happened |
|---|---|
| v01 -> v02 | **Increased efficiency** I started with a general geodesic solver that computed Christoffel symbols numerically, so it worked for any diagonal metric. However, this was significantly slower in practice as most symbols computed are 0s. I then derived the equations of motion by hand from the Euler-Lagrange equations, which was significantly faster (30x speedup). |
| v01 -> v02 | **Negative universe rays not rendered** Rays going through the throat had negative $l$, so they were not caught by the escape check ($l > 40$) and thus rendered black. This was fixed by checking $\lvert l \rvert$. |
| v03 -> v04 | **Camera stuck in one universe.** Originally, the camera travelled between universes by flipping the sign of the radius based on the x coordinate (a hack really). This was fixed by setting the camera in wormhole coordinates directly. |
| After v04 | **Rays leaving at the wrong angles.** Previously, I used the camera's direction components directly as angular velocities for the rays without dividing by the local radius. So the throat was scaled to the wrong size. This was fixed in the final version. |

Full version history: [versions/](versions/)

## Limitations

- This is an Ellis-Bronnikov wormhole which is the simplest form of more general Morris-Thorne wormholes. It has no gravity outside the throat while more general wormholes can also attract light and matter.
- The sky is placed on a sphere at $|l| = 40$ rather than at true infinity which can cause a slight parallax effect.
- 360° frames may show small artefacts directly above and below the camera (at the pole $\theta = 0$) where the spherical coordinates break down.
- 32-bit floating values are used for speed limiting precision to ~10⁻⁵.
- The camera moves along a single straight line through the throat.


## References

Ellis (1973); Bronnikov (1973); Morris & Thorne (1988); James et al. (2015, *Visualizing Interstellar's Wormhole*); Dormand & Prince (1980).

**Star map:** NASA/Goddard Space Flight Center Scientific Visualization Studio, *Deep Star Maps 2020*. Gaia DR2: ESA/Gaia/DPAC. Constellation figures based on those developed for the IAU by Alan MacRobert of Sky and Telescope (Roger Sinnott and Rick Fienberg).

**Second sky:** ESO/S. Brunier, The Milky Way panorama (eso0932a), CC BY 4.0.