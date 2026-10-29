"""
Ellis wormhole ray tracer
=========================
Renders what a camera would see near and while travelling through an Ellis–Bronnikov traversable wormhole

l is the proper radial distance, l > 0 and l < 0 are two separate universes joined at
the throat (l = 0, radius b0) and each has its own sky image

Light rays are traced backwards from the camera along null geodesics. The accelerations
were derived by hand from the Euler–Lagrange equations and integrated with the same
adaptive Dormand–Prince RK45 integrator as the Kerr ray tracer, on the GPU with Taichi

Usage:   python ellis_wormhole.py
  MODE = "image"  -> one frame from l = CAMERA_L, saved to render.png
  MODE = "video"  -> camera flies through the throat, frames saved to frames/

Sky images (optional) in assets/: starmap_8k.exr (l < 0) and eso_milky_way.tif (l > 0).
Without them, default differently coloured checkerboards are used

Units: geometric units, G = c = 1
"""


import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" #OpenCV disables EXR reading by default so environment variable must be set before importing cv2

import numpy as np
import taichi as ti
import time
import cv2

ti.init(arch=ti.gpu) # default CUDA on NVIDIA gpus, otherwise Vulkan/Metal

# ─────────────────────────── Settings ───────────────────────────
# Wormhole
throat_radius = 5.0      # b0: radius of the throat at l = 0
sky_distance = 40.0      # sky_distance sampling the universe's sky

# Integrator (step sizes are in the affine parameter)
tol = 1e-6               # RK45 tolerance
dt_start = 1e-3
dt_min = 1e-10
dt_max = 10.0
steps = 20000            # maximum RK45 steps per ray (rays that never escape are shown magenta)

# Output
MODE = "video"           # "image" or "video"
width, height = 2880, 1440 #Resolution

VR_360 = False            # 360 degrees equirectangular frames, requires a 2:1 resolution, e.g. 3840x1920
FOV = 110                # horizontal field of view in degrees (ignored when VR_360 is on)
FPS = 30
duration = 30            # video length in seconds
total_frames = FPS * duration

# Camera (moves along the x-axis: theta = pi/2, phi = 0)
camera_dir = ti.Vector([1.0, 0.0, 0.0])   # direction of increasing l
CAMERA_L = -10.0         # image mode: camera position (negative = starmap side)
l_start = -2.5      # video mode: camera path through the throat
l_end = 2.5






# Ray state (0 = still travelling, 1 - escaped)
ESCAPED = 1

offsets = ti.Vector.field(n=3, dtype=ti.f32, shape=5)

pos = ti.Vector.field(4, dtype=ti.f32, shape=(5, height, width))   
vel = ti.Vector.field(4, dtype=ti.f32, shape=(5, height, width))   

state = ti.field(dtype=ti.i32, shape=(5, height, width))
dt    = ti.field(dtype=ti.f32, shape=(5, height, width))

colours = ti.Vector.field(n=3, shape=(5, height, width), dtype=ti.f32)
avg_colours = ti.Vector.field(n=3, shape=(height, width), dtype=ti.f32)


@ti.func
def calculate_derivatives(x, u):

    # State x = (t, l, theta, phi) and u = dx/dlambda, with lambda = affine parameter
    # Accelerations derived by hand from the Euler–Lagrange equations of the Ellis metric
    # (faster than computing Christoffel symbols numerically):
    # g_tt = -1 everywhere, so d^2t/dlambda^2 = 0 (no gravitational redshift)

    l = x[1]
    theta = x[2]

    l_dot = u[1]
    theta_dot = u[2]
    phi_dot = u[3]

    sin_theta = ti.math.sin(theta)
    cos_theta = ti.math.cos(theta)
    R_sq = l**2 + throat_radius**2   # (areal radius)^2

    accel = ti.Vector([
        0.0,
        l * (theta_dot**2 + sin_theta**2 * phi_dot**2),
        sin_theta * cos_theta * phi_dot**2 - 2.0 * l * l_dot * theta_dot / R_sq,
        -2.0 * cos_theta / ti.max(1e-5, ti.abs(sin_theta)) * theta_dot * phi_dot - 2.0 * l * l_dot * phi_dot / R_sq,
    ])

    return u, accel


# Dormand prince RK-45 method
@ti.func
def calculate_new_pos(i, j, k):
    x1 = pos[i, j, k]
    u1 = vel[i, j, k]
    step = dt[i, j, k]

    step_new = step
    x7 = ti.Vector([0.0, 0.0, 0.0, 0.0])
    u7 = ti.Vector([0.0, 0.0, 0.0, 0.0])

    for iteration in range(50): #max count so no infinite loops
        d_x1, d_u1 = calculate_derivatives(x1, u1) #k1

        x2 = x1 + d_x1 * step/5
        u2 = u1 + d_u1 * step/5

        d_x2, d_u2 = calculate_derivatives(x2, u2)

        x3 = x1 + (3*d_x1 + 9*d_x2) * step/40
        u3 = u1 + (3*d_u1 + 9*d_u2) * step/40

        d_x3, d_u3 = calculate_derivatives(x3, u3)

        x4 = x1 + (44/45 * d_x1 - 56/15 * d_x2 + 32/9*d_x3) * step
        u4 = u1 + (44/45 * d_u1 - 56/15 * d_u2 + 32/9*d_u3) * step

        d_x4, d_u4 = calculate_derivatives(x4, u4)

        x5 = x1 + (19372/6561 * d_x1 - 25360/2187 * d_x2 + 64448/6561 * d_x3 - 212/729 * d_x4) * step
        u5 = u1 + (19372/6561 * d_u1 - 25360/2187 * d_u2 + 64448/6561 * d_u3 - 212/729 * d_u4) * step

        d_x5, d_u5 = calculate_derivatives(x5, u5)

        x6 = x1 + (9017/3168 * d_x1 - 355/33 * d_x2 + 46732/5247 * d_x3 + 49/176 * d_x4 - 5103/18656 * d_x5) * step
        u6 = u1 + (9017/3168 * d_u1 - 355/33 * d_u2 + 46732/5247 * d_u3 + 49/176 * d_u4 - 5103/18656 * d_u5) * step

        d_x6, d_u6 = calculate_derivatives(x6, u6)

        #5th order solutions
        x7 = x1 + (35/384 * d_x1 + 500/1113 * d_x3 + 125/192 * d_x4 - 2187/6784 * d_x5 + 11/84 * d_x6) * step
        u7 = u1 + (35/384 * d_u1 + 500/1113 * d_u3 + 125/192 * d_u4 - 2187/6784 * d_u5 + 11/84 * d_u6) * step

        d_x7, d_u7 = calculate_derivatives(x7, u7)

        # Error estimated from difference between 5th and 4th order solutions which is directly computed
        x_error = step * ti.abs(71/57600 * d_x1 - 71/16695 * d_x3 + 71/1920 * d_x4 - 17253/339200 * d_x5 + 22/525 * d_x6 - 1/40 * d_x7)
        u_error = step * ti.abs(71/57600 * d_u1 - 71/16695 * d_u3 + 71/1920 * d_u4 - 17253/339200 * d_u5 + 22/525 * d_u6 - 1/40 * d_u7)

        error = 0.0
        for index in ti.static(range(4)):
            error = ti.max(error, ti.max(x_error[index], u_error[index]))

        if error < tol:
            if error > 0:
                step_new = 0.9 * step * (tol / error) ** 0.2 #if error is less than tolerance, step size scales up otherwise it shrinks down
            else:
                step_new = 2.0 * step

            step_new = max(step / 5.0, min(step * 5.0, step_new))
            step_new = max(dt_min, min(dt_max, step_new))
            break
        else:
            step = 0.9 * step * (tol / error) ** 0.25
            if step < dt_min:
                step_new = dt_min
                break

    return x7, u7, step_new


@ti.func
def update(i, j, k):

    pos_new, vel_new, dt_new = calculate_new_pos(i, j, k)

    pos[i, j, k] = pos_new
    vel[i, j, k] = vel_new
    dt[i, j, k] = dt_new

    check_collided(i, j, k)


@ti.func
def check_collided(i, j, k):
    l = pos[i, j, k][1]

    if ti.abs(l) > sky_distance:
        state[i, j, k] = ESCAPED


@ti.kernel
def inits(cam_l: ti.f32, cam_theta: ti.f32, cam_phi: ti.f32, camera_dir: ti.types.vector(3, ti.f32), FOV: ti.f32):

    fov_x = FOV * ti.math.pi / 180.0
    fov_y = fov_x / (width / height)

    pixel_width = fov_x / width
    pixel_height = fov_y / height

    diff_x = pixel_width * 0.35
    diff_y = pixel_height * 0.35

    offsets[0] = ti.Vector([0.0, 0.0, 0.0])
    offsets[1] = ti.Vector([0.0, diff_x, 0.0])
    offsets[2] = ti.Vector([0.0, -diff_x, 0.0])
    offsets[3] = ti.Vector([0.0, 0.0, diff_y])
    offsets[4] = ti.Vector([0.0, 0.0, -diff_y])

    # Camera basis: forward, right and up unit vectors
    forward = ti.math.normalize(camera_dir)
    fake_up = ti.Vector([0.0, 0.0, 1.0]) #used to obtain vector perp to forward
    if ti.abs(forward[2]) > 0.999:
        fake_up = ti.Vector([0.0, 1.0, 0.0]) #make sure forward not parallel to fakeup
    right = ti.math.normalize(ti.math.cross(forward, fake_up))
    up = ti.math.normalize(ti.math.cross(right, forward)) #actual up vector

    # Orthonormal frame at the camera (e_l, e_theta, e_phi), the same for every ray
    e_l = ti.Vector([
        ti.math.sin(cam_theta)*ti.math.cos(cam_phi),
        ti.math.sin(cam_theta)*ti.math.sin(cam_phi),
        ti.math.cos(cam_theta)
    ])

    e_theta = ti.Vector([
        ti.math.cos(cam_theta)*ti.math.cos(cam_phi),
        ti.math.cos(cam_theta)*ti.math.sin(cam_phi),
        -ti.math.sin(cam_theta)
    ])

    e_phi = ti.Vector([
        -ti.math.sin(cam_phi),
        ti.math.cos(cam_phi),
        0.0
    ])

    R = ti.math.sqrt(cam_l**2 + throat_radius**2)   # areal radius at the camera

    for j, i, k in ti.ndrange(5, height, width):

        dx = k / (width - 1.0) * 2.0 - 1.0
        dy = i / (height - 1.0) * 2.0 - 1.0 #range from -1 to 1

        init_yaw = dx * (fov_x / 2.0)
        init_pitch = dy * (fov_y / 2.0) #Using angular coords for each pixel

        if VR_360:
            init_yaw = dx * ti.math.pi
            init_pitch = -dy * (ti.math.pi / 2.0)

        yaw = init_yaw + offsets[j][1]
        pitch = init_pitch + offsets[j][2] #offsets for anti aliasing

        dir_local = ti.Vector([ti.math.cos(yaw) * ti.math.cos(pitch), ti.math.sin(yaw) * ti.math.cos(pitch), ti.math.sin(pitch)]) #convert sphere to vector

        rotation_matrix = ti.Matrix.cols([forward, right, up]) #matrix to obtain actual vx,vy,vz

        ray_dir = ti.math.normalize(rotation_matrix @ dir_local) #matrix vector multiplication

        # Components of the ray direction in the camera's orthonormal frame
        v_l = ti.math.dot(ray_dir, e_l)
        v_theta = ti.math.dot(ray_dir, e_theta)
        v_phi = ti.math.dot(ray_dir, e_phi)

        pos[j, i, k] = ti.Vector([0.0, cam_l, cam_theta, cam_phi])

        # Convert to coordinate velocities using g_ll = 1, g_θθ = R², g_φφ = R² sin²θ
        # (null condition: -1 + v_l^2 + v_theta^2 + v_phi^2 = 0)
        vel[j, i, k] = ti.Vector([
            1.0,                                      
            v_l,                                      
            v_theta / R,                              
            v_phi / (R * ti.math.sin(cam_theta)),     
        ])

        state[j, i, k] = 0
        dt[j, i, k] = dt_start


@ti.kernel
def render_rays():
    for i, j, k in ti.ndrange(5, height, width):
        for step in range(steps):
            if state[i, j, k] != 0:
                break

            l_prev = pos[i, j, k][1]
            theta_prev = pos[i, j, k][2]
            phi_prev = pos[i, j, k][3]

            update(i, j, k)

            if state[i, j, k] == ESCAPED: #interpolation for accurate sky sampling
                l_new = pos[i, j, k][1]
                theta_new = pos[i, j, k][2]
                phi_new = pos[i, j, k][3]

                edge = sky_distance if l_new > 0 else -sky_distance
                fraction = (edge - l_prev) / (l_new - l_prev + 1e-10)

                pos[i, j, k][1] = edge
                pos[i, j, k][2] = theta_prev + fraction * (theta_new - theta_prev)
                pos[i, j, k][3] = phi_prev + fraction * (phi_new - phi_prev)



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(SCRIPT_DIR, "assets")

def load_hdri(path, tint):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Sky image not found at {path}, using a checkerboard sky. "
              "See README for the downloads.")
        h, w = 1024, 2048
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((((xx // 64) + (yy // 64)) % 2) * 0.5).astype(np.float32)
        img = checker[..., None] * np.array(tint, dtype=np.float32)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype == np.uint8:
            img = (img.astype(np.float32) / 255.0) ** 2.2      # 8-bit sRGB -> linear
        elif img.dtype == np.uint16:
            img = (img.astype(np.float32) / 65535.0) ** 2.2    # 16-bit sRGB -> linear
        else:
            img = img.astype(np.float32)                       # EXR/HDR: already linear
        
    h, w, c = img.shape
    ti_img = ti.Vector.field(c, dtype=ti.f32, shape=(h, w))
    ti_img.from_numpy(img)
    return ti_img


@ti.func
def sample_hdri(l, theta, phi):
    # The sign of l selects the universe: l > 0 -> sky_positive, l < 0 -> sky_negative
    if l > 0:
        theta = ti.math.pi - theta
        phi = phi + ti.math.pi

    theta = theta % (2*ti.math.pi) #wrap angles into range
    if theta > ti.math.pi:
        theta = 2 * ti.math.pi - theta
        phi += ti.math.pi
    phi = (phi + ti.math.pi) % (2 * ti.math.pi) - ti.math.pi

    u = 0.5 + phi / (2 * ti.math.pi)
    v = 1.0 - (theta / ti.math.pi)

    colour = ti.Vector([0.0, 0.0, 0.0])
    if l > 0:
        x = int(u * (sky_positive.shape[1] - 1)) % sky_positive.shape[1]
        y = int(v * (sky_positive.shape[0] - 1)) % sky_positive.shape[0]
        colour = sky_positive[y, x]
    else:
        x = int(u * (sky_negative.shape[1] - 1)) % sky_negative.shape[1]
        y = int(v * (sky_negative.shape[0] - 1)) % sky_negative.shape[0]
        colour = sky_negative[y, x]
    return colour

sky_negative = load_hdri(os.path.join(ASSETS, "starmap_8k.exr"), tint=(1.0, 1.0, 1.0))  # l < 0
sky_positive = load_hdri(os.path.join(ASSETS, "eso_milky_way.tif"),        tint=(1.0, 0.6, 0.2))  # l > 0


@ti.func
def render_colours():
    # The wormhole has no horizon, so every ray should escape to one of the two skies
    for i, j, k in ti.ndrange(5, height, width):
        s = state[i, j, k]

        r = 0.0
        g = 0.0
        b = 0.0
        if s == ESCAPED:
            c_sky = sample_hdri(pos[i, j, k][1], pos[i, j, k][2], pos[i, j, k][3])

            r = c_sky[0]
            g = c_sky[1]
            b = c_sky[2]
        else: # ray didn't escape within steps - debug colour (magenta)
            r = 1.0
            g = 0.0
            b = 1.0

        #Reinhard Tone mapping - compresses light pixels but leaves dark pixels alone for the screen
        r_map = r / (r + 1.0)
        g_map = g / (g + 1.0)
        b_map = b / (b + 1.0)

        #gamma correction - makes image look more natural to human eye when displayed
        colours[i, j, k][0] = r_map **(1/2.2)
        colours[i, j, k][1] = g_map **(1/2.2)
        colours[i, j, k][2] = b_map **(1/2.2)


@ti.kernel
def process_image():
    render_colours()

    weight_centre = 0.5
    weight_sides = 0.125

    for j, k in ti.ndrange(height, width):
        color_sum = ti.Vector([0.0, 0.0, 0.0])
        for i in range(5):
            weight = weight_centre if i == 0 else weight_sides
            color_sum += colours[i, j, k] * weight
        avg_colours[j, k] = color_sum


def save_image(filename="render.png"):
    process_image()
    image = avg_colours.to_numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #cv2 uses bgr instead of rgb
    cv2.imwrite(filename, image_bgr)


def update_camera(l):
    # Straight path along the x-axis (theta = pi/2, phi = 0), always facing increasing l
    return l, np.pi / 2, 0.0, camera_dir


def generate_video():
    os.makedirs("frames", exist_ok=True) #create frames

    print("Initializing")
    start_time = time.perf_counter()
    frame = 0

    while frame < total_frames:
        cam_l, cam_theta, cam_phi, cam_dir = update_camera(l_start + (l_end - l_start) * frame / total_frames)

        inits(cam_l, cam_theta, cam_phi, cam_dir, FOV)
        render_rays()
        ti.sync()

        name = f"frames/frame_{frame:04d}.png"
        save_image(filename=name)

        if frame % 20 == 0:
            print(f"Frame {frame} / {total_frames}")
        frame += 1

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f}")
    print(f"Stitch with: ffmpeg -framerate {FPS} -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    if MODE == "image":
        cam_l, cam_theta, cam_phi, cam_dir = update_camera(CAMERA_L)
        inits(cam_l, cam_theta, cam_phi, cam_dir, FOV)
        render_rays()
        ti.sync()
        save_image("render.png")
        print("Saved render.png")
    else:
        generate_video()
