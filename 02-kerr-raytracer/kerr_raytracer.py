'''
Ray tracing simulation of a Kerr (spinning) black hole
======================================================
This code renders what a camera near a spinning black hole would see-
the black hole's shadow, gravitationally lensed light from stars, and a glowing volumetric
Novikov–Thorne accretion disk

Every pixel fires 5 rays (for anti-aliasing)
Each ray is traced backwards from the camera along a null geodesic of the Kerr metric
(in Boyer–Lindquist coordinates, Mino time)
using an adaptive Dormand–Prince RK45 integrator, in parallel on the GPU with Taichi

Usage:   python kerr_raytracer.py

  MODE = "image"  - renders one frame to render.png
  MODE = "video"  - renders the camera path in update_camera() to frames/

Star map (optional): save NASA SVS's "Deep Star Maps 2020" (8K EXR) as assets/starmap_8k.exr
Without it, a default checkerboard sky is used as the background

Units: geometric units, G = c = M = 1

'''


import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" #OpenCV disables EXR reading by default, so the environment variable must be set before importing cv2

import numpy as np
import taichi as ti
import time
import cv2

ti.init(arch=ti.gpu) # default CUDA on NVIDIA gpus, otherwise Vulkan/Metal

# ─────────────────────────── Settings ───────────────────────────
# Black hole and disk
a = 0.5                  # spin: 0 = schwarzschild, approaching 1 - maximal Kerr spin
accretion_disk = True
accretion_disk_r = 20    # disk extends from the ISCO out to ISCO + this radius (only prograde)

# Integrator (step sizes are in Mino time)
tol = 1e-6               # RK45 tolerance: 1e-6 ~ 1–2 s/frame, 1e-8 ~ 60 s/frame, removes faint artefacts in disk
dt_start = 1e-3
dt_min = 1e-10
dt_max = 10.0
steps = 20000            # maximum RK45 steps per ray
radius = 100              # rays reaching this radius sample the star map, set 50 to shorten render time

# Output
MODE = "image"           # "image" or "video"
width, height = 1920, 1080 #Resolution
# GPU memory ~ 370 bytes per pixel + star map - around 1.2 GB at 1080p, and around 3.5 GB at 4K

VR_360 = False           # 360 degrees equirectangular frames, requires a 2:1 resolution, e.g. 3840x1920
FOV = 110                # horizontal field of view in degrees (ignored when VR_360 is on)
FPS = 30
duration = 30            # video length in seconds
total_frames = FPS * duration






# Ray states (0 = still travelling)
HORIZON, DISK_OPAQUE, ESCAPED = 1, 2, 3

offsets = ti.Vector.field(n=3, dtype=ti.f32, shape=5)

pos   = ti.Vector.field(3, dtype=ti.f32, shape=(5,height,width))
velo  = ti.Vector.field(3, dtype=ti.f32, shape=(5,height,width))
pvelo = ti.Vector.field(2, dtype=ti.f32, shape=(5,height,width))

L_z           = ti.field(dtype=ti.f32,shape=(5,height,width))
Q             = ti.field(dtype=ti.f32,shape=(5,height,width))
state         = ti.field(dtype=ti.i32,shape=(5,height,width))
dt            = ti.field(dtype=ti.f32,shape=(5,height,width))
transmittance = ti.field(dtype=ti.f32,shape=(5,height,width))
intensity     = ti.field(dtype=ti.f32,shape=(5,height,width))
energy        = ti.field(dtype=ti.f32,shape=(5,height,width))

colours = ti.Vector.field(n=3,shape=(5,height,width),dtype=ti.f32)
avg_colours = ti.Vector.field(n=3,shape=(height,width),dtype=ti.f32)

# Helper functions
@ti.func
def outer_event_horizon_R(a):
     return 1 + ti.math.sqrt(ti.max(1 - a**2, 0.0))

def ISCO(a): #Innermost stable circular orbit, prograde with M = 1
     Z1 = 1 + ((1 - a**2) ** (1/3)) * ( (1 + a)**(1/3) + (1 - a)**(1/3) )
     Z2 = np.sqrt(3*a**2 + Z1**2)

     r_ISCO = 3 + Z2 - np.sqrt( (3 - Z1) * (3 + Z1 + 2*Z2) )
     return r_ISCO

@ti.func
def to_spherical(pos):
    w2 = pos[0]**2 + pos[1]**2 + pos[2]**2
    r = ti.math.sqrt(ti.abs(w2 - a**2 + ti.math.sqrt(ti.abs((w2 - a**2)**2 + 4 * a**2 * pos[2]**2))) / 2)
    theta = ti.acos(ti.min(ti.max(pos[2] / r, -1.0), 1.0)) #clip to avoid errors
    phi = ti.atan2(pos[1],pos[0])
    return r,theta,phi

@ti.func
def to_cartesian(pos):
    r = pos[0]
    theta = pos[1]
    phi = pos[2]
    
    x = ti.math.sqrt(ti.max(r**2+a**2,0.0)) * ti.math.sin(theta) * ti.math.cos(phi)
    y = ti.math.sqrt(ti.max(r**2+a**2,0.0)) * ti.math.sin(theta) * ti.math.sin(phi)
    z = r * ti.math.cos(theta)
    return x,y,z

@ti.func
def calculate_derivatives(r, theta, phi, pr, ptheta, i,j,k):
     
     # State (r, theta, phi) and pr = dr/dlambda, ptheta = d(theta)/dlambda, with lambda = Mino time
     # r and theta use second-order equations, so turning points don't need manual sign flips
     # phi has no turning points thus stays first order

     Q_val = Q[i,j,k]
     L_z_val = L_z[i,j,k]

     sin_theta = ti.max(1e-5,ti.abs(ti.math.sin(theta)))
     cos_theta = ti.math.cos(theta)
     
     d_pr = 2*r*(r**2 + a**2 - a*L_z_val) - (r - 1)*(Q_val + (L_z_val - a)**2)
     d_ptheta = cos_theta * (L_z_val**2 / (sin_theta**3) - a**2 * sin_theta)

     d_r = pr
     d_theta = ptheta
     d_phi = a / ((r**2 + a**2 - 2*r) + 1e-10) * ((r**2 + a**2) - a*L_z_val) - a + L_z_val / (sin_theta**2)

     return d_r, d_theta, d_phi, d_pr, d_ptheta

# Dormand prince RK-45 method
@ti.func
def calculate_new_pos(i,j,k):
     r = pos[i,j,k][0]
     theta = pos[i,j,k][1]
     phi = pos[i,j,k][2]
     pr = pvelo[i,j,k][0]
     ptheta = pvelo[i,j,k][1]
     step = dt[i,j,k]
     
     step_new = step
     r7 = theta7 = phi7 = pr7 = ptheta7 = 0.0
     
     for iteration in range(50): #max count so no infinite loops
         d_r1, d_theta1, d_phi1, d_pr1, d_ptheta1 = calculate_derivatives(r, theta, phi, pr, ptheta,i,j,k) #k1
         
         r2 = r + d_r1 * step/5
         theta2 = theta + d_theta1 * step/5
         phi2 = phi + d_phi1 * step/5
         pr2 = pr + d_pr1 * step/5
         ptheta2 = ptheta + d_ptheta1 * step/5

         d_r2, d_theta2, d_phi2, d_pr2, d_ptheta2 = calculate_derivatives(r2, theta2, phi2, pr2, ptheta2, i,j,k)

         r3 = r + (3*d_r1 + 9*d_r2) * step/40
         theta3 = theta + (3*d_theta1 + 9*d_theta2) * step/40
         phi3 = phi + (3*d_phi1 + 9*d_phi2) * step/40
         pr3 = pr + (3*d_pr1 + 9*d_pr2) * step/40
         ptheta3 = ptheta + (3*d_ptheta1 + 9*d_ptheta2) * step/40

         d_r3, d_theta3, d_phi3, d_pr3, d_ptheta3 = calculate_derivatives(r3, theta3, phi3, pr3, ptheta3, i,j,k)
          
         r4 = r + (44/45 * d_r1 - 56/15 * d_r2 + 32/9*d_r3) * step
         theta4 = theta + (44/45 * d_theta1 - 56/15 * d_theta2 + 32/9*d_theta3) * step
         phi4 = phi + (44/45 * d_phi1 - 56/15 * d_phi2 + 32/9*d_phi3) * step
         pr4 = pr + (44/45 * d_pr1 - 56/15 * d_pr2 + 32/9*d_pr3) * step
         ptheta4 = ptheta + (44/45 * d_ptheta1 - 56/15 * d_ptheta2 + 32/9*d_ptheta3) * step

         d_r4, d_theta4, d_phi4, d_pr4, d_ptheta4 = calculate_derivatives(r4, theta4, phi4, pr4, ptheta4, i,j,k)

         r5 = r + (19372/6561 * d_r1 - 25360/2187 * d_r2 + 64448/6561 * d_r3 - 212/729 * d_r4) * step
         theta5 = theta + (19372/6561 * d_theta1 - 25360/2187 * d_theta2 + 64448/6561 * d_theta3 - 212/729 * d_theta4) * step
         phi5 = phi + (19372/6561 * d_phi1 - 25360/2187 * d_phi2 + 64448/6561 * d_phi3 - 212/729 * d_phi4) * step
         pr5 = pr + (19372/6561 * d_pr1 - 25360/2187 * d_pr2 + 64448/6561 * d_pr3 - 212/729 * d_pr4) * step
         ptheta5 = ptheta + (19372/6561 * d_ptheta1 - 25360/2187 * d_ptheta2 + 64448/6561 * d_ptheta3 - 212/729 * d_ptheta4) * step

         d_r5, d_theta5, d_phi5, d_pr5, d_ptheta5 = calculate_derivatives(r5, theta5, phi5, pr5, ptheta5, i,j,k)

         r6 = r + (9017/3168 * d_r1 - 355/33 * d_r2 + 46732/5247 * d_r3 + 49/176 * d_r4 - 5103/18656 * d_r5) * step
         theta6 = theta + (9017/3168 * d_theta1 - 355/33 * d_theta2 + 46732/5247 * d_theta3 + 49/176 * d_theta4 - 5103/18656 * d_theta5) * step
         phi6 = phi + (9017/3168 * d_phi1 - 355/33 * d_phi2 + 46732/5247 * d_phi3 + 49/176 * d_phi4 - 5103/18656 * d_phi5) * step
         pr6 = pr + (9017/3168 * d_pr1 - 355/33 * d_pr2 + 46732/5247 * d_pr3 + 49/176 * d_pr4 - 5103/18656 * d_pr5) * step
         ptheta6 = ptheta + (9017/3168 * d_ptheta1 - 355/33 * d_ptheta2 + 46732/5247 * d_ptheta3 + 49/176 * d_ptheta4 - 5103/18656 * d_ptheta5) * step

         d_r6, d_theta6, d_phi6, d_pr6, d_ptheta6 = calculate_derivatives(r6, theta6, phi6, pr6, ptheta6, i,j,k)


         #5th order solutions
         r7 = r + (35/384 * d_r1 + 500/1113 * d_r3 + 125/192 * d_r4 - 2187/6784 * d_r5 + 11/84 * d_r6) * step
         theta7 = theta + (35/384 * d_theta1 + 500/1113 * d_theta3 + 125/192 * d_theta4 - 2187/6784 * d_theta5 + 11/84 * d_theta6) * step
         phi7 = phi + (35/384 * d_phi1 + 500/1113 * d_phi3 + 125/192 * d_phi4 - 2187/6784 * d_phi5 + 11/84 * d_phi6) * step
         pr7 = pr + (35/384 * d_pr1 + 500/1113 * d_pr3 + 125/192 * d_pr4 - 2187/6784 * d_pr5 + 11/84 * d_pr6) * step
         ptheta7 = ptheta + (35/384 * d_ptheta1 + 500/1113 * d_ptheta3 + 125/192 * d_ptheta4 - 2187/6784 * d_ptheta5 + 11/84 * d_ptheta6) * step

         d_r7, d_theta7, d_phi7, d_pr7, d_ptheta7 = calculate_derivatives(r7, theta7, phi7, pr7, ptheta7, i,j,k)

         # Error estimated from difference between 5th and 4th order solutions which is directly computed
         r_error = step * ti.abs(71/57600 * d_r1 - 71/16695 * d_r3 + 71/1920 * d_r4 - 17253/339200 * d_r5 + 22/525 * d_r6 - 1/40 * d_r7)
         theta_error = step * ti.abs(71/57600 * d_theta1 - 71/16695 * d_theta3 + 71/1920 * d_theta4 - 17253/339200 * d_theta5 + 22/525 * d_theta6 - 1/40 * d_theta7)
         phi_error = step * ti.abs(71/57600 * d_phi1 - 71/16695 * d_phi3 + 71/1920 * d_phi4 - 17253/339200 * d_phi5 + 22/525 * d_phi6 - 1/40 * d_phi7)
         pr_error = step * ti.abs(71/57600 * d_pr1 - 71/16695 * d_pr3 + 71/1920 * d_pr4 - 17253/339200 * d_pr5 + 22/525 * d_pr6 - 1/40 * d_pr7)
         ptheta_error = step * ti.abs(71/57600 * d_ptheta1 - 71/16695 * d_ptheta3 + 71/1920 * d_ptheta4 - 17253/339200 * d_ptheta5 + 22/525 * d_ptheta6 - 1/40 * d_ptheta7)

         error = ti.max(r_error, ti.max(theta_error, ti.max(phi_error, ti.max(pr_error, ptheta_error))))

         if error < tol:
              if error > 0:
                   step_new = 0.9 * step * (tol / error) ** 0.2 #if error is less than tolerance, step size scales up, else it shrinks down
              else:
                   step_new = 2.0 * step

              step_new = max(step / 5.0, min(step * 5.0, step_new))     
              step_new = max(dt_min,min(dt_max,step_new))
              break
         else:
              step = 0.9 * step * (tol / error) ** 0.25
              if step < dt_min:
                  step_new = dt_min
                  break

     return r7,theta7,phi7,pr7,ptheta7,step_new
  

@ti.func
def update(i,j,k,r_ISCO):
     
    dt_val = dt[i,j,k]
    
    r_new,theta_new,phi_new,pr_new,ptheta_new,dt_new = calculate_new_pos(i,j,k)
    pos_new = ti.Vector([r_new,theta_new,phi_new])
    pvelo_new = ti.Vector([pr_new,ptheta_new])

    pos[i,j,k] = pos_new
    pvelo[i,j,k] = pvelo_new
    dt[i,j,k] = dt_new
    
    check_collided(i,j,k,dt_val,r_ISCO)


#Some components of the metric needed below:
@ti.func
def g_tt(r):
     return -1.0 * (1 - 2 / r)
@ti.func
def g_tphi(r,a):
     return -2.0 * a / r
@ti.func
def g_phiphi(r,a):
     return (r ** 2) + (a ** 2) + 2 * (a ** 2) / r


@ti.func
def Q_f(x,a,x_ISCO):
     B_val = B_f(x,a)
     term_1 = (2.0 * (x - x_ISCO) - 3 * a * ti.math.log(x/x_ISCO)) / (2.0 * B_val)

     #Finding the 3 roots of the polynomial x^3 - 3x + 2a = 0
     x1 = 2.0 * ti.math.cos(ti.acos(a) / 3.0 - ti.math.pi / 3.0)
     x2 = 2.0 * ti.math.cos(ti.acos(a) / 3.0 + ti.math.pi / 3.0)
     x3 = -2.0 * ti.math.cos(ti.acos(a) / 3.0)
     xn = ti.Vector([x1,x2,x3]) #storing in a vector

     sum_term = 0.0
     for i in range(3):
          j = ((i+1) % 3)
          k = ((i+2) % 3) #handling the other indices apart from i
          if ti.abs(xn[i]) > 1e-6:
               sum_term += (3*(xn[i] - a)**2 * ti.math.log((x - xn[i])/(x_ISCO - xn[i]))) / (xn[i] * B_val * (xn[i] - xn[j]) * (xn[i] - xn[k]))
          
     return term_1 - sum_term




#Page–Thorne relativistic correction factors
@ti.func
def B_f(x,a):                        
     return 1.0 + a * (x ** -3)

@ti.func
def C_f(x,a):
     return 1 - 3 * (x ** -2) + 2 * a * (x ** -3)

#Accounts for kinematic doppler shift and gravitational redshift
@ti.func
def redshift_factor(r, b, i, j, k):

    x = ti.math.sqrt(r)
    omega = 1.0 / (x**3 + a)   #angular velocity of prograde gas
    time_dilation = ti.math.sqrt(-1 * g_tt(r) - 2 * omega * g_tphi(r, a) - omega**2 * g_phiphi(r, a))
    g_inf = time_dilation / (1.0 - omega * b)
    g_obs = g_inf / energy[i, j, k]
    return g_inf, g_obs

@ti.func
def vertical_density(r,z,r_ISCO):
     
     n0 = 1 # Baseline optical thickness at inner edge
     H0 = 0.2 # Scale height factor - increasing makes it more puffy
     p = 1.125 # Flaring Power index - determines flaring at upper/lower edges of disk
     fade_edge = 10 #disk starts fading
     n_mid = n0 * (r / r_ISCO) ** -1.5
     H = H0 * r_ISCO * (r / r_ISCO) ** p #gaussian vertical profile
     density  = n_mid * ti.math.exp(-z**2/(2*H**2))
     
     if r > accretion_disk_r + r_ISCO - fade_edge:

          t = ti.max(0,ti.min(1,(accretion_disk_r+ r_ISCO-r)/fade_edge ))

          smooth_factor = t * t * (3.0 - 2.0 * t) #Using the smoothstep function to smoothen out the abrupt edge
          density *= smooth_factor
     
     return density


@ti.func
def radial_temperature(r, r_ISCO):
     M_dot = 3 # Mass accretion rate - determines the glow linearly
     T = 0.0
     if r >= r_ISCO - 1e-5:
         # Relativistic Novikov–Thorne temperature with Page–Thorne correction factors
         r = ti.max(r,r_ISCO+1e-2)
         x = ti.math.sqrt(r)
         x_ISCO = ti.math.sqrt(r_ISCO)
         T = (M_dot/(r**3) * (Q_f(x,a,x_ISCO) / (B_f(x,a) * ti.math.sqrt(C_f(x,a))))) ** 0.25
     return T
     
@ti.func
def source(T):
     EXPOSURE = 2000.0 # brightness scale (displaying choice, not a physical effect)
     S = EXPOSURE * T**4
     return S


@ti.func
def volumetric_calculations(i, j, k, r,theta,phi, z, r_ISCO, delta_s, g_obs):

    if transmittance[i,j,k] > 1e-5:

         # Using the Novikov-Thorne Model
         absorption_coeff = 0.025 # Absorption coefficient - Control how opaque the disk is
         t = absorption_coeff * vertical_density(r,z,r_ISCO) * delta_s #opacity
         T = ti.math.exp(-t) #Transmission

         Temp = radial_temperature(r, r_ISCO)
         S_i = source(Temp)
         E_i = S_i * (1 - T)

         I_new = intensity[i,j,k] + (E_i * transmittance[i,j,k]) * (g_obs ** 4)
         Transm_new = transmittance[i,j,k] * T
         transmittance[i,j,k] = Transm_new
         intensity[i,j,k] = I_new

         if Transm_new < 1e-4:
              state[i,j,k] = DISK_OPAQUE

@ti.func
def check_collided(i,j,k,dt_val,r_ISCO):
    disk_thickness=0.8
    r = pos[i,j,k][0]
    theta = pos[i,j,k][1]
    phi = pos[i,j,k][2]
    z = r * ti.math.cos(theta)

    horizon = outer_event_horizon_R(a) + 0.01

    if r <= horizon:
        state[i,j,k] = HORIZON
        
    elif accretion_disk and (ti.abs(z) <= disk_thickness / 2.0) and (r > r_ISCO) and (r < r_ISCO+accretion_disk_r): 

         Sigma = r**2 + a**2 * ti.math.cos(theta)**2
         g_inf, g_obs = redshift_factor(r, L_z[i, j, k], i, j, k)
         delta_s = (Sigma * dt_val) / g_inf

         volumetric_calculations(i, j, k, r,theta,phi, z, r_ISCO, delta_s, g_obs)
         
    if (r > radius):
         state[i,j,k] = ESCAPED


@ti.kernel
def inits(init_camera_pos: ti.types.vector(3, ti.f32), camera_dir: ti.types.vector(3, ti.f32), FOV: ti.f32):

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
    fake_up = ti.Vector([0.0,0.0,1.0]) #used to obtain vector perp to forward
    if ti.abs(forward[2]) > 0.999:
         fake_up = ti.Vector([0.0,1.0,0.0]) #make sure forward not parallel to fakeup
    right = ti.math.normalize(ti.math.cross(forward,fake_up))
    up = ti.math.normalize(ti.math.cross(right, forward)) #actual up vector

    for j,i,k in ti.ndrange(5,height,width):

          pos[j,i,k] = init_camera_pos

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

          rotation_matrix = ti.Matrix.cols([forward,right,up]) #matrix to obtain actual vx,vy,vz 

          ray_dir = rotation_matrix @ dir_local #matrix vector multiplication 

          velo[j,i,k] = ti.math.normalize(ray_dir)

          p = pos[j,i,k]
          v = velo[j,i,k]

          r,theta,phi = to_spherical(p)

          #Local basis vectors
          e_r = ti.Vector([
             ti.math.sin(theta)*ti.math.cos(phi), 
             ti.math.sin(theta)*ti.math.sin(phi), 
             ti.math.cos(theta)
          ])

          e_theta = ti.Vector([
             ti.math.cos(theta)*ti.math.cos(phi), 
             ti.math.cos(theta)*ti.math.sin(phi), 
             -ti.math.sin(theta)
          ])

          e_phi = ti.Vector([
             -ti.math.sin(phi), 
             ti.math.cos(phi), 
             0.0
          ])

          
          #Local velocities
          v_r = ti.math.dot(v,e_r)
          v_theta = ti.math.dot(v,e_theta)
          v_phi = ti.math.dot(v,e_phi)

          #Kerr Metric Components
          Sigma = r**2 + a**2 * ti.math.cos(theta)**2
          Delta = r**2 + a**2 - 2.0*r
          A = (r**2 + a**2)**2 - a**2 * Delta * ti.math.sin(theta)**2

          #Calculating the covariant momentum from the local velocity (ZAMO tetrad transformation)
          p_r_cov = ti.math.sqrt(Sigma / Delta) * v_r
          p_theta_cov = ti.math.sqrt(Sigma) * v_theta
          p_phi_cov = ti.math.sqrt(A / Sigma) * ti.math.sin(theta) * v_phi

          omega = 2.0 * a * r / A
          E_photon = ti.math.sqrt(Delta * Sigma / A) * 1.0 + omega * p_phi_cov

          L_z_init = p_phi_cov / E_photon
          pr_init = Delta * (p_r_cov / E_photon)
          ptheta_init = p_theta_cov / E_photon
          
          Q_init = ptheta_init**2 + ti.math.cos(theta)**2 * (L_z_init**2 / (ti.math.sin(theta)**2 + 1e-10) - a**2)

          L_z[j,i,k] = L_z_init
          Q[j,i,k] = Q_init
          pvelo[j,i,k][0] = pr_init
          pvelo[j,i,k][1] = ptheta_init

          pos[j,i,k][0] = r
          pos[j,i,k][1] = theta
          pos[j,i,k][2] = phi

          transmittance[j,i,k] = 1.0
          intensity[j,i,k] = 0.0
          state[j,i,k] = 0
          dt[j,i,k] = dt_start
          energy[j,i,k] = E_photon

r_ISCO = ISCO(a)

@ti.kernel
def render_rays():
     for i,j,k in ti.ndrange(5,height,width):
          for step in range(steps):
              if state[i,j,k] != 0:
                 break

              r_prev = pos[i,j,k][0]
              theta_prev = pos[i,j,k][1]
              phi_prev = pos[i,j,k][2]
              
              update(i,j,k,r_ISCO)

              state_new = state[i,j,k]
              r_new = pos[i,j,k][0]
              theta_new = pos[i,j,k][1]
              phi_new = pos[i,j,k][2]

              if state_new == ESCAPED : #interpolation for accurate sky sampling
                   fraction = (radius - r_prev) / (r_new - r_prev + 1e-10)

                   pos[i,j,k][0] = radius
                   pos[i,j,k][1] = theta_prev + fraction * (theta_new - theta_prev)
                   pos[i,j,k][2] = phi_prev + fraction * (phi_new - phi_prev)



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
def sample_hdri(theta,phi):
    theta = theta % (2*ti.math.pi) #safety first - wrap angles into range
    if theta > ti.math.pi:
        theta = 2 * ti.math.pi - theta
        phi += ti.math.pi
    phi = (phi + ti.math.pi) % (2 * ti.math.pi) - ti.math.pi
    
    u = 0.5 + phi / (2 * ti.math.pi)
    v = 1.0 - (theta / ti.math.pi)

    height = hdri_map.shape[0]
    width = hdri_map.shape[1] 
    x = int(u * (width - 1)) % width
    y = int(v * (height - 1)) % height

    return hdri_map[y, x]

hdri_map = load_hdri(STARMAP_PATH)

@ti.func
def render_colours():
      # Rays hitting the horizon remain black
     for i,j,k in ti.ndrange(5,height,width):     
                    s = state[i,j,k]
                    
                    r = 0.0
                    g = 0.0
                    b = 0.0
                    if s == ESCAPED:
                         theta = pos[i,j,k][1]
                         phi = pos[i,j,k][2]
                         c_sky = sample_hdri(theta,phi)

                         star_shift = (1.0 / energy[i,j,k]) ** 4 # light from distant stars blueshifted by 1/E as it falls into the gravity well to the camera
                         
                         r = c_sky[0] * transmittance[i,j,k] * star_shift
                         g = c_sky[1] * transmittance[i,j,k] * star_shift
                         b = c_sky[2] * transmittance[i,j,k] * star_shift

                    disk_intensity = intensity[i,j,k]
                    r += disk_intensity * 0.60 
                    g += disk_intensity * 0.70
                    b += disk_intensity * 1.50 #strong blue hint

                    #Reinhard Tone mapping - compresses light pixels but leaves dark pixels alone for the screen
                    r_map = r / (r + 1.0)
                    g_map = g / (g + 1.0)
                    b_map = b / (b + 1.0)
                    
                    #gamma correction - makes image look more natural to human eye when displayed
                    colours[i,j,k][0] = r_map **(1/2.2)
                    colours[i,j,k][1] = g_map **(1/2.2)
                    colours[i,j,k][2] = b_map **(1/2.2)
          
@ti.kernel
def process_image():
     render_colours()

     weight_centre = 0.5
     weight_sides = 0.125
     
     for j,k in ti.ndrange(height,width):
          color_sum = ti.Vector([0.0, 0.0, 0.0])
          for i in range(5):
               weight = weight_centre if i == 0 else weight_sides
               color_sum += colours[i,j,k] * weight
          avg_colours[j,k] = color_sum


def save_image(filename="render.png"):
    process_image()
    image = avg_colours.to_numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #cv2 uses bgr instead of rgb
    cv2.imwrite(filename, image_bgr)

init_t = -20.0
final_t = -1.8

def update_camera(t):
     global FOV
     
     #follow parametric trajectory
     x = t
     y = t
     z = 0.5 #straight line toward the black hole

     init_camera_pos = ti.Vector([x,y,z])

     distance = np.sqrt(x**2 + y**2 + z**2)
     FOV = 110.0
     pan_angle = 0.0

     start_dist = 6.0
     if (not VR_360) and distance < start_dist:
          factor = max(0.0, min(1.0, ((start_dist - distance) / (start_dist - 2.0) )))
          smooth_factor = factor * factor * (3.0 - 2.0 * factor) #Using the smoothstep function again to smoothen out the abrupt edge
          
          pan_angle = 90 * smooth_factor
          FOV = 110.0 + (175.0 - 110.0) * factor
     pan_angle = np.deg2rad(pan_angle)

     #establishing local coordinates at camera to determine panning direction (same as in inits)
     np_pos = np.array([x,y,z])
     forward = -np_pos / distance
     
     fake_up = np.array([0.0,0.0,1.0]) #used to obtain vector perp to forward
     if np.abs(forward[2]) > 0.999:
         fake_up = np.array([0.0,1.0,0.0])
     right = np.cross(forward,fake_up)
     right = right / np.linalg.norm(right)

     new_dir = forward * np.cos(pan_angle) + right * np.sin(pan_angle) #pans to side based on how close camera is to EH
     camera_dir = ti.Vector(new_dir.tolist())

     return init_camera_pos,camera_dir

def generate_video():
     os.makedirs("frames", exist_ok=True) #create frames
     
     print("Initializing")
     start_time = time.perf_counter()
     frame = 0
     
     while frame < total_frames:
          init_camera_pos,camera_dir = update_camera(init_t + (final_t - init_t) * frame / total_frames)
          
          inits(init_camera_pos,camera_dir,FOV)
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
        inits(init_camera_pos, camera_dir, FOV)
        render_rays()
        ti.sync()
        save_image("render.png")
        print("Saved render.png")
    else:
        generate_video()
          
          
     
     
     


