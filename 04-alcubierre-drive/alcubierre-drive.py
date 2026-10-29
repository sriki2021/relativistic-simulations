'''
Ray tracing simulation of an Alcubierre warp drive
==================================================
This code renders what a stationary observer would see as an Alcubierre warp bubble
travels by them faster than light
The ship is drawn as a sphere at the bubble's centre, coloured by the time its light left it

Every pixel fires 5 rays (for anti-aliasing)
Each ray is traced backwards in time from the camera along a null geodesic. The
accelerations were derived by hand from the Euler–Lagrange equations and integrated with
an adaptive Dormand–Prince RK45 integrator, in parallel on the GPU with Taichi

Usage:   python alcubierre_warp_drive.py

  MODE = "image"  - renders one frame at t = init_t to render.png
  MODE = "video"  - renders frames from t = init_t to final_t into frames/

Star map (optional): save NASA SVS's "Deep Star Maps 2020" (8K EXR) as assets/starmap_8k.exr
Without it a default checkerboard sky is used as the background

Units: geometric units, G = c = 1 (v_0 = 4 -> four times the speed of light)

'''


import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" #OpenCV disables EXR reading by default, so the environment variable must be set before importing cv2

import numpy as np
import taichi as ti
import time
import cv2

ti.init(arch=ti.gpu) # default CUDA on NVIDIA gpus, otherwise Vulkan/Metal

# ─────────────────────────── Settings ───────────────────────────
# Warp bubble
v_0 = 4.0                # bubble speed at t = 0 in units of c (more than 1 -> faster than light)
a_s = 0.0                # bubble acceleration
R = 1.5                  # bubble radius
sigma = 1.5              # wall sharpness, wall thickness ~ 1/sigma
ship_radius = 1.4        # radius of the ship sphere at the bubble's centre
SHIP_COLOUR_T_MIN, SHIP_COLOUR_T_MAX = -4.7, 2.35   # emission times shown from blue to red on the ship

# Integrator (step sizes are in the affine parameter)
tol = 1e-6               # RK45 tolerance
dt_start = 1e-3
dt_min = 1e-10
dt_max = 20.0
steps = 20000            # maximum RK45 steps per ray
STALL_LIMIT = 16         # consecutive minimum-size steps before a ray is marked as stalled

# Output
MODE = "image"           # "image" or "video"
width, height = 1920, 1080 #Resolution

VR_360 = False           # 360 degrees equirectangular frames, requires a 2:1 resolution e.g. 3840x1920
FOV = 120                # horizontal field of view in degrees (ignored when VR_360 is on)
FPS = 60
duration = 30            # video length in seconds
total_frames = FPS * duration






# Ray states (0 = still travelling)
TRAPPED, SHIP, ESCAPED, STALLED = 1, 2, 3, 4

offsets = ti.Vector.field(n=3, dtype=ti.f32, shape=5)

pos = ti.Vector.field(4, dtype=ti.f32, shape=(5, height, width))   # (t, x, y, z)
vel = ti.Vector.field(4, dtype=ti.f32, shape=(5, height, width))   # d(pos)/d(lambda)

state = ti.field(dtype=ti.i32, shape=(5, height, width))
dt    = ti.field(dtype=ti.f32, shape=(5, height, width))
stall = ti.field(dtype=ti.i32, shape=(5, height, width))           # consecutive steps at dt_min

colours = ti.Vector.field(n=3, shape=(5, height, width), dtype=ti.f32)
avg_colours = ti.Vector.field(n=3, shape=(height, width), dtype=ti.f32)


# Helper functions
@ti.func
def bubble_position(t):
    return v_0 * t + 0.5 * a_s * t**2

@ti.func
def bubble_speed(t):
    return v_0 + a_s * t

@ti.func
def sech(x):
    return 2.0 / (ti.math.exp(x) + ti.math.exp(-x))

@ti.func
def shape_function(r_s):#  _
    # Alcubierre's shape _/ \_ : ~1 inside the bubble, ~0 outside
    return (ti.math.tanh(sigma * (r_s + R)) - ti.math.tanh(sigma * (r_s - R))) / (2.0 * ti.math.tanh(sigma * R))

@ti.func
def shape_function_derivative(r_s):
    return sigma * (sech(sigma * (r_s + R))**2 - sech(sigma * (r_s - R))**2) / (2.0 * ti.math.tanh(sigma * R))


@ti.func
def calculate_derivatives(q, u):

    # State q (t, x, y, z) and u = dq/dlambda, with lambda = affine parameter
    # Accelerations derived by hand from the Euler Lagrange equations of motion
    t_coord = q[0]
    dx_s = q[1] - bubble_position(t_coord)   # x distance from the bubble's centre
    y = q[2]
    z = q[3]

    t_dot = u[0]
    x_dot = u[1]
    y_dot = u[2]
    z_dot = u[3]

    v_s = bubble_speed(t_coord)
    r_s = ti.math.sqrt(dx_s**2 + y**2 + z**2)

    f = shape_function(r_s)
    df_dr = shape_function_derivative(r_s)

    dr_dx = dx_s / (r_s + 1e-10)
    dr_dy = y / (r_s + 1e-10)
    dr_dz = z / (r_s + 1e-10)

    D = v_s * f
    D_x = v_s * df_dr * dr_dx
    D_y = v_s * df_dr * dr_dy
    D_z = v_s * df_dr * dr_dz                # chain rule through r_s
    D_t = a_s * f - v_s**2 * df_dr * dr_dx   # the bubble moves so r_s depends on t

    p_x = x_dot - D * t_dot
    D_dot = D_t * t_dot + D_x * x_dot + D_y * y_dot + D_z * z_dot

    t_acc = -p_x * (D_x * p_x + D_y * y_dot + D_z * z_dot)

    accel = ti.Vector([
        t_acc,
        -D_x * t_dot * p_x + D_dot * t_dot + D * t_acc,
        -D_y * t_dot * p_x,
        -D_z * t_dot * p_x,
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
    forced = 0     # 1 once the step has reached dt_min
    accepted = 0

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

        if error < tol or forced == 1:
            accepted = 1
            if error > 0:
                step_new = 0.9 * step * (tol / error) ** 0.2 #if error is less than tolerance, step size scales up, else it shrinks down
            else:
                step_new = 2.0 * step

            # Limit the next step near the ship so that a ray cannot jump over it
            r_s_now = ti.math.sqrt((x1[1] - bubble_position(x1[0]))**2 + x1[2]**2 + x1[3]**2)
            cap = ti.min(dt_max, ti.max(0.02, 0.15 * (r_s_now - ship_radius)))

            step_new = max(step / 5.0, min(step * 5.0, step_new))
            step_new = max(dt_min, min(cap, step_new))
            break
        else:
            step = 0.9 * step * (tol / error) ** 0.25
            if step < dt_min:
                step = dt_min   # retry once at the minimum step size and accept it
                forced = 1

    if accepted == 0:
        step_new = dt_min   # 50 rejections in a row - try again small on the next step

    # Checks if the step passed through the ship
    A = ti.Vector([(x7[1] - bubble_position(x7[0])) - (x1[1] - bubble_position(x1[0])), x7[2] - x1[2], x7[3] - x1[3]])
    B = ti.Vector([x1[1] - bubble_position(x1[0]), x1[2], x1[3]])

    aa = A.dot(A) + 1e-12
    ab = A.dot(B)
    cc = B.dot(B) - ship_radius**2
    disc = ab*ab - aa*cc

    if disc > 0.0:
        u_hit = (-ab - ti.math.sqrt(disc)) / aa
        if u_hit >= 0.0 and u_hit <= 1.0:
            state[i, j, k] = SHIP
            x7 = x1 + (x7 - x1) * u_hit   # move to the point where the ray meets the ship

    return x7, u7, step_new


@ti.func
def update(i, j, k):

    pos_new, vel_new, dt_new = calculate_new_pos(i, j, k)

    pos[i, j, k] = pos_new
    vel[i, j, k] = vel_new
    dt[i, j, k] = dt_new

    # rays stuck at the minimum step size for too long are demarcated stalled
    if dt_new <= dt_min * 1.0001:
        stall[i, j, k] += 1
    else:
        stall[i, j, k] = 0

    if stall[i, j, k] >= STALL_LIMIT and state[i, j, k] == 0:
        state[i, j, k] = STALLED

    check_collided(i, j, k)


@ti.func
def check_collided(i, j, k):
    q = pos[i, j, k]
    u = vel[i, j, k]

    x_comoving = q[1] - bubble_position(q[0])        # x' = x - x_s(t)
    x_comoving_dot = u[1] - bubble_speed(q[0]) * u[0]

    if state[i, j, k] == 0:   # don't overwrite a ship hit or a stall
        if x_comoving > R + 4.0 / sigma:
            state[i, j, k] = ESCAPED
        elif x_comoving_dot < 1e-3 * ti.abs(u[0]):
            state[i, j, k] = TRAPPED


@ti.kernel
def inits(cam_t: ti.f32, init_camera_pos: ti.types.vector(3, ti.f32), camera_dir: ti.types.vector(3, ti.f32), FOV: ti.f32):

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

        pos[j, i, k] = ti.Vector([cam_t, init_camera_pos[0], init_camera_pos[1], init_camera_pos[2]])

        # Traced backwards in time (dt/dlambda = -1). The camera is outside the bubble, where
        # spacetime is flat (f ~ 0), so a unit spatial direction makes the ray light-like
        vel[j, i, k] = ti.Vector([-1.0, ray_dir[0], ray_dir[1], ray_dir[2]])

        state[j, i, k] = 0
        dt[j, i, k] = dt_start
        stall[j, i, k] = 0


@ti.kernel
def render_rays():
    # No boundary interpolation is needed: escaped rays sample the sky by their final direction
    for i, j, k in ti.ndrange(5, height, width):
        for step in range(steps):
            if state[i, j, k] != 0:
                break

            update(i, j, k)



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STARMAP_PATH = os.path.join(SCRIPT_DIR, "assets", "starmap_8k.exr")

def load_hdri(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Star map not found at {path}, using a checkerboard sky. "
              "See README for the NASA download.")
        h, w = 1024, 2048
        yy, xx = np.mgrid[0:h, 0:w]
        img = ((((xx // 64) + (yy // 64)) % 2) * 0.5).astype(np.float32)
        img = np.repeat(img[..., None], 3, axis=2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w, c = img.shape
    ti_img = ti.Vector.field(c, dtype=ti.f32, shape=(h, w))
    ti_img.from_numpy(img)
    return ti_img


@ti.func
def sample_hdri(theta, phi):
    theta = theta % (2*ti.math.pi) #wrap angles into range
    if theta > ti.math.pi:
        theta = 2 * ti.math.pi - theta
        phi += ti.math.pi
    phi = (phi + ti.math.pi) % (2 * ti.math.pi) - ti.math.pi

    u = 0.5 + phi / (2 * ti.math.pi)
    v = 1.0 - (theta / ti.math.pi)

    x = int(u * (hdri_map.shape[1] - 1)) % hdri_map.shape[1]
    y = int(v * (hdri_map.shape[0] - 1)) % hdri_map.shape[0]

    return hdri_map[y, x]

hdri_map = load_hdri(STARMAP_PATH)


@ti.func
def render_colours():
    for i, j, k in ti.ndrange(5, height, width):
        s = state[i, j, k]

        r = 0.0
        g = 0.0
        b = 0.0
        if s == ESCAPED:
            # The sky is at infinity, so it is sampled by ray's final direction
            direction = ti.math.normalize(ti.Vector([vel[i, j, k][1], vel[i, j, k][2], vel[i, j, k][3]]))
            theta = ti.acos(ti.min(ti.max(direction[2], -1.0), 1.0))
            phi = ti.atan2(direction[1], direction[0])
            c_sky = sample_hdri(theta, phi)

            r = c_sky[0]
            g = c_sky[1]
            b = c_sky[2]
        elif s == SHIP:
            # Colour shows when the light left the ship - blue (early) to red (late)
            age = ti.min(1.0, ti.max(0.0, (pos[i, j, k][0] - SHIP_COLOUR_T_MIN) / (SHIP_COLOUR_T_MAX - SHIP_COLOUR_T_MIN)))
            r = age
            g = 0.15
            b = 1.0 - age
        else:
            # Trapped by the bubble's horizon, stalled, or out of steps: dark purple
            r = 0.03
            g = 0.0
            b = 0.09

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


init_t = 9.2
final_t = 17.5

def update_camera(t):
    # static camera beside the bubble's path (x-axis) facing the path and turned a bit
    init_camera_pos = ti.Vector([0.0, -12.0, 0.0])
    camera_dir = ti.Vector([-0.1650, 0.9863, 0.0])
    return init_camera_pos, camera_dir


def check_camera_outside_bubble(t, cam_pos):
    # The rays only start light-like if the camera is in flat space, outside the bubble wall
    x_s = v_0 * t + 0.5 * a_s * t**2
    r_s = np.sqrt((cam_pos[0] - x_s)**2 + cam_pos[1]**2 + cam_pos[2]**2)
    if r_s < R + 4.0 / sigma:
        print(f"[WARN] At t = {t:.2f} the camera is inside or near the bubble wall, rays will not start light-like")


def generate_video():
    os.makedirs("frames", exist_ok=True) #create frames

    print("Initializing")
    start_time = time.perf_counter()
    frame = 0

    while frame < total_frames:
        t = init_t + (final_t - init_t) * frame / total_frames
        init_camera_pos, camera_dir = update_camera(t)
        check_camera_outside_bubble(t, init_camera_pos)

        inits(t, init_camera_pos, camera_dir, FOV)
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
        init_camera_pos, camera_dir = update_camera(init_t)
        check_camera_outside_bubble(init_t, init_camera_pos)
        inits(init_t, init_camera_pos, camera_dir, FOV)
        render_rays()
        ti.sync()
        save_image("render.png")
        print("Saved render.png")
    else:
        generate_video()
