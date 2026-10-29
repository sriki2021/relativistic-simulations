import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" #idk why it was disabled

import numpy as np
import taichi as ti
import time
import cv2

ti.init(arch=ti.cuda) #using nvidia gpu

# Suppress warnings for invalid math in the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

# best render options: tol = 7-8, 4K res, 

#Changeables

radius = 50 #radius of world
b = 5.0     # minimum throat radius
dt_start = 0.001
dt_min = 1e-10 #e-10 orig -8 fine
dt_max = 10.0
tol = 1e-6 # 1e-8: 60s, Stunning (no tiny artefacts whatsoever), 1e-6: 2s best choice

eps = 1e-4 #size in calcing derivatives

#CAMERA VARS

width = 3840
height = 2160

##Resolutions (width x height):
##
##360p:   640 x 360
##720p:   1280 x 720
##1080p:  1920 x 1080
##1440p:  2560 x 1440
##4K UHD: 3840 x 2160 or 4096
##Beyond this can't be saved to memory

VR_360 = False #Resolution must be 2:1 ratio

FOV = 110
FPS = 30
duration = 30
total_frames = FPS * duration


aliasing = True

init_camera_pos = ti.Vector([0.0, 5.0, 2.0])
camera_dir =  -1*init_camera_pos














#Parameters
steps = 20000

offsets = ti.Vector.field(n=3, dtype=ti.f32, shape=5)
camera_pos = ti.Vector.field(n=3,dtype=ti.f32,shape=(5,))

num_rays = 5 * width * height #num photons

pos   = ti.Vector.field(4, dtype=ti.f32, shape=(5,height,width))
v  = ti.Vector.field(4, dtype=ti.f32, shape=(5,height,width))
#pvelo = ti.Vector.field(2, dtype=ti.f32, shape=(5,height,width))


state         = ti.field(dtype=ti.i32,shape=(5,height,width))
dt            = ti.field(dtype=ti.f32,shape=(5,height,width))
##transmittance = ti.field(dtype=ti.f32,shape=(5,height,width))
##intensity     = ti.field(dtype=ti.f32,shape=(5,height,width))
##energy        = ti.field(dtype=ti.f32,shape=(5,height,width))

colours = ti.Vector.field(n=3,shape=(5,height,width),dtype=ti.f32)
avg_colours = ti.Vector.field(n=3,shape=(height,width),dtype=ti.f32)

@ti.func
def to_spherical(pos):
    r = ti.math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    theta = ti.acos(ti.min(ti.max(pos[2] / r, -1.0), 1.0)) #clip to avoid errors
    phi = ti.atan2(pos[1],pos[0])
    return r,theta,phi

@ti.func
def metric(m,n,pos): #Ellis Bronnikov metric
    gmn = 0.0
    if m == n:
         if m == 0:
              gmn = -1.0
         elif m == 1:
              gmn = 1.0
         elif m == 2:
              gmn = pos[1]**2 + b**2
         elif m == 3:
              gmn = (pos[1]**2 + b**2) * ti.math.sin(pos[2])**2
    return gmn

@ti.func
def del_metric(index, m,n,eps,pos): #use better algorithm
     pos_plus = pos
     pos_plus[index] += eps
     pos_minus = pos
     pos_minus[index] -= eps

     return (metric(m,n,pos_plus) - metric(m,n,pos_minus)) / (2.0 * eps)

@ti.func
def inverse_metric(m,n,pos):    #ONLY works for diagonal metrics
    g_inv = 0.0
    if m == n:
        g_inv = 1.0 / metric(m,n,pos)
    return g_inv

@ti.func
def christoffel(k,i,j, pos): #index k up, i,j down
     sum_terms = 0.0
##     for l in range(4):
##          sum_derivatives = del_metric(j, i,l,eps,pos) + del_metric(i, j,l,eps,pos) - del_metric(l, i,j,eps,pos)
##          
##          sum_terms += 1/2.0 * inverse_metric(k,l,pos) * sum_derivatives
     inv_g = inverse_metric(k,k,pos)
     if inv_g != 0.0:
         if k == i:
             sum_terms += del_metric(j, k,i,eps,pos)
         if k == j:
             sum_terms += del_metric(i, k,j,eps,pos)
         if i == j:
             sum_terms -= del_metric(k, i,j,eps,pos)
         sum_terms *= 0.5 * inv_g
     return sum_terms
             
     return sum_terms

@ti.func
def acceleration(i, v, pos): #Solves the geodesic equation for acceleration of x_i from velocity vector v
     sum_terms = 0.0
     for j in ti.static(range(4)):
          for k in ti.static(range(4)):
               sum_terms += - christoffel(i,j,k,pos) * v[j] * v[k]

     return sum_terms

@ti.func
def calculate_derivatives(pos, v):
     
    #Used the Lagrangian formulation and Euler-Lagrange equations to calculate exactly by hand formulae for the accelerations (much faster than christoffel)

     a = ti.Vector([0.0,0.0,0.0,0.0])
     d_x = ti.Vector([0.0,0.0,0.0,0.0])

     r = pos[1]
     theta = pos[2]
     phi = pos[3]

     r_dot = v[1]
     theta_dot = v[2]
     phi_dot = v[3]

     sin_theta = ti.math.sin(theta)
     cos_theta = ti.math.cos(theta)

     a[0] = 0.0
     a[1] = r * (theta_dot**2 + sin_theta**2 * phi_dot**2)
     a[2] = sin_theta * cos_theta * phi_dot**2 - 2.0 * r * r_dot * theta_dot / (r**2 + b**2)
     a[3] = - 2.0 * cos_theta / (ti.max(sin_theta, 1e-7)) * theta_dot * phi_dot - 2.0 * r * r_dot * phi_dot / (r**2 + b**2)

     for i in ti.static(range(4)):
         d_x[i] = v[i]

     return d_x, a

# NEW dormand prince RK-45 method :O
@ti.func
def calculate_new_pos(i,j,k): 

     x1 = pos[i,j,k]
     v1 = v[i,j,k]
     step = dt[i,j,k]
     step_new = step
     x7 = ti.Vector([0.0, 0.0, 0.0, 0.0])
     v7 = ti.Vector([0.0, 0.0, 0.0, 0.0])

     iteration = 0

     while iteration < 50: #max count so no infinite loops
          iteration += 1
          
          d_x1, d_v1 = calculate_derivatives(x1,v1)

          x2 = x1 + d_x1 * step/5
          v2 = v1 + d_v1 * step/5

          d_x2, d_v2 = calculate_derivatives(x2,v2)

          x3 = x1 + (3*d_x1 + 9*d_x2) * step/40
          v3 = v1 + (3*d_v1 + 9*d_v2) * step/40

          d_x3, d_v3 = calculate_derivatives(x3,v3)

          x4 = x1 + (44/45 * d_x1 - 56/15 * d_x2 + 32/9*d_x3) * step
          v4 = v1 + (44/45 * d_v1 - 56/15 * d_v2 + 32/9*d_v3) * step

          d_x4, d_v4 = calculate_derivatives(x4,v4)
          
          x5 = x1 + (19372/6561 * d_x1 - 25360/2187 * d_x2 + 64448/6561 * d_x3 - 212/729 * d_x4) * step
          v5 = v1 + (19372/6561 * d_v1 - 25360/2187 * d_v2 + 64448/6561 * d_v3 - 212/729 * d_v4) * step

          d_x5, d_v5 = calculate_derivatives(x5,v5)
          
          x6 = x1 + (9017/3168 * d_x1 - 355/33 * d_x2 + 46732/5247 * d_x3 + 49/176 * d_x4 - 5103/18656 * d_x5) * step
          v6 = v1 + (9017/3168 * d_v1 - 355/33 * d_v2 + 46732/5247 * d_v3 + 49/176 * d_v4 - 5103/18656 * d_v5) * step

          d_x6, d_v6 = calculate_derivatives(x6,v6)
          
          x7 = x1 + (35/384 * d_x1 + 500/1113 * d_x3 + 125/192 * d_x4 - 2187/6784 * d_x5 + 11/84 * d_x6) * step          
          v7 = v1 + (35/384 * d_v1 + 500/1113 * d_v3 + 125/192 * d_v4 - 2187/6784 * d_v5 + 11/84 * d_v6) * step

          d_x7, d_v7 = calculate_derivatives(x7,v7)

          x_error = step * ti.abs(71/57600 * d_x1 - 71/16695 * d_x3 + 71/1920 * d_x4 - 17253/339200 * d_x5 + 22/525 * d_x6 - 1/40 * d_x7)
          v_error = step * ti.abs(71/57600 * d_v1 - 71/16695 * d_v3 + 71/1920 * d_v4 - 17253/339200 * d_v5 + 22/525 * d_v6 - 1/40 * d_v7)

          error = 0.0
          for index in ti.static(range(4)):
               if x_error[index] > error: error = x_error[index]
               if v_error[index] > error: error = v_error[index]

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
                  #r7 = 0.0
                  step_new = dt_min
                  break

     return x7,v7,step_new
### Consider passing dr_7 etc so that it can be reused

@ti.func
def update(i,j,k):
    
    pos_new,v_new,dt_new = calculate_new_pos(i,j,k)

    pos[i,j,k] = pos_new
    v[i,j,k] = v_new
    dt[i,j,k] = dt_new
    
    checkCollided(i,j,k)

@ti.func
def checkCollided(i,j,k):
    r = pos[i,j,k][1]

    if (ti.abs(r) > radius):
         state[i,j,k] = 3
    


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
     
    # camera vectors fun
    #obtain the 3 unit vectors relative to camera pos: forward up and right


    forward = ti.math.normalize(camera_dir)
    fake_up = ti.Vector([0.0,0.0,1.0]) #used to obtain vector perp to forward
    if ti.abs(forward[2]) > 0.999:
         fake_up = ti.Vector([0.0,1.0,0.0]) #make sure forward not parallel to fakeup
    right = ti.math.normalize(ti.math.cross(forward,fake_up))
    up = ti.math.normalize(ti.math.cross(right, forward)) #actual up vector


    for j,i,k in ti.ndrange(5,height,width):

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

          velo = ti.math.normalize(ray_dir)

          p = init_camera_pos
          r,theta,phi = to_spherical(p)

          #Local basis vectors for the given metric
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
          v_r = ti.math.dot(velo,e_r)
          v_theta = ti.math.dot(velo,e_theta)
          v_phi = ti.math.dot(velo,e_phi)

          pos[j,i,k][0] = 0.0
          pos[j,i,k][1] = r
          pos[j,i,k][2] = theta
          pos[j,i,k][3] = phi

          v[j,i,k][0] = 1.0
          v[j,i,k][1] = v_r
          v[j,i,k][2] = v_theta
          v[j,i,k][3] = v_phi

          


          state[j,i,k] = 0
          dt[j,i,k] = dt_start

@ti.kernel
def render_rays():
     for i,j,k in ti.ndrange(5,height,width):
          for step in range(steps):
              if state[i,j,k] != 0:
                 break

              r_prev = pos[i,j,k][1]
              theta_prev = pos[i,j,k][2]
              phi_prev = pos[i,j,k][3]
              
              update(i,j,k)

              state_new = state[i,j,k]
              r_new = pos[i,j,k][1]
              theta_new = pos[i,j,k][2]
              phi_new = pos[i,j,k][3]

##              if state_new == 3: #interpolation for smoothification
##                   fraction = (radius - r_prev) / (r_new - r_prev + 1e-10)
##
##                   pos[i,j,k][1] = radius
##                   pos[i,j,k][2] = theta_prev + fraction * (theta_new - theta_prev)
##                   pos[i,j,k][3] = phi_prev + fraction * (phi_new - phi_prev)


def load_hdri(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load HDRI file at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    h,w,c = img.shape
    ti_img = ti.Vector.field(c,dtype=ti.f32,shape=(h,w))
    ti_img.from_numpy(img)
    return ti_img

@ti.func
def sample_hdri(i,theta,phi):
    theta = theta % (2*ti.math.pi) #safety first
    if theta > ti.math.pi:
        theta = 2 * ti.math.pi - theta
        phi += ti.math.pi
    phi = (phi + ti.math.pi) % (2 * ti.math.pi) - ti.math.pi
    
    u_coord = 0.5 + phi / (2 * ti.math.pi)
    v_coord = 1.0 - (theta / ti.math.pi)

    height_hdri = hdri_map1.shape[0]
    width_hdri = hdri_map1.shape[1]
    if i == 2:
        height_hdri = hdri_map2.shape[0]
        width_hdri = hdri_map2.shape[1]
    x = ti.cast(u_coord * (width_hdri - 1),ti.i32) % width_hdri
    y = ti.cast(v_coord * (height_hdri - 1),ti.i32) % height_hdri
    
    mapping = hdri_map1[y, x]
    if i == 2:
        mapping = hdri_map2[y, x]
    return mapping

hdri_map1 = load_hdri("starmap_8k.exr")
hdri_map2 = load_hdri("sky.hdr")

@ti.func
def render_colours():
      #default is landing in event horizon
     for i,j,k in ti.ndrange(5,height,width):     
                    # 1 - Horizon, 2 - Accretion Disk, 3 - Escaper
                    s = state[i,j,k]
                    
                    r = 0.0
                    g = 0.0
                    b = 0.0
                    if s == 3:
                         radial = pos[i,j,k][1]
                         theta = pos[i,j,k][2]
                         phi = pos[i,j,k][3]
                         
                         c_sky = sample_hdri(1,theta,phi)
                         if radial < 0:
                             c_sky = sample_hdri(2,theta,phi)

                         r = c_sky[0] 
                         g = c_sky[1]
                         b = c_sky[2]

##                    disk_intensity = intensity[i,j,k]
##                    exposure = 1.0 #debugging super darkness
##                    r += disk_intensity * 0.60 * exposure
##                    g += disk_intensity * 0.70 * exposure
##                    b += disk_intensity * 1.50 * exposure #strong blue hint

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
     
     #if aliasing:
     weight_centre = 0.5
     weight_sides = 0.125
     
     for j,k in ti.ndrange(height,width):
          color_sum = ti.Vector([0.0, 0.0, 0.0])
          for i in range(5):
               weight = weight_centre if i == 0 else weight_sides
               color_sum += colours[i,j,k] * weight
          avg_colours[j,k] = color_sum


def print_image(filename="balck.png"):
    process_image()
    image = avg_colours.to_numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #cv2 uses bgr instead of rgb
    cv2.imwrite(filename, image_bgr)

def simulate_photons():
    print("helo")
    start_time = time.perf_counter()
    print("Initializing...")
    inits(init_camera_pos,camera_dir,FOV)
    print("Raytracing on GPU...")
    render_rays()
    ti.sync()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f}, Resolution: {width} x {height}, dt: {dt_start}")
    print_image()
    print("Render saved")
    

simulate_photons()


# orbit at constant r
##     orbit_rad = 15.0
##     z_height = 2.0
##     
##     init_camera_pos = ti.Vector([orbit_rad*np.sin(angle), orbit_rad*np.cos(angle), z_height])
##     camera_dir =  -1*init_camera_pos

#ti.Vector([7.0 - t, np.sqrt(t), np.log(t+1)])

##     x = 20.0 * np.exp(-0.05*t) * np.sin(t)
##     y = -0.5 + 7.0 * np.exp(-0.05*t)
##     z = 20.0 * np.exp(-0.05*t) * np.cos(t) #spiral
##
##init_t = -20.0
##final_t = 0.0
##
##def update_camera(t):
##     global init_camera_pos,camera_dir,FOV
##     
##     #follow parametric trajectory
##     x = t
##     y = t
##     z = 0.5 #parabola
##
##     init_camera_pos = ti.Vector([x,y,z])
##
##     distance = np.sqrt(x**2 + y**2 + z**2)
##     FOV = 110.0
##     pan_angle = 0.0
##
##     start_dist = 6.0
##     if (not VR_360) and distance < start_dist:
##          factor = max(0.0, min(1.0, ((start_dist - distance) / (start_dist - 2.0) )))
##          smooth_factor = factor * factor * (3.0 - 2.0 * factor) #Using the smoothstep function again to smoothen out the abrupt edge
##          
##          pan_angle = 90 * smooth_factor
##          FOV = 110.0 + (175.0 - 110.0) * factor
##     pan_angle = np.deg2rad(pan_angle)
##
##     #establishing local coordinates at camera to determine panning direction (same as in inits)
##     np_pos = np.array([x,y,z])
##     forward = -np_pos / distance
##     
##     fake_up = np.array([0.0,0.0,1.0]) #used to obtain vector perp to forward
##     if np.abs(forward[2]) > 0.999:
##         fake_up = np.array([0.0,1.0,0.0])
##     right = np.cross(forward,fake_up)
##     right = right / np.linalg.norm(right)
##
##     new_dir = forward * np.cos(pan_angle) + right * np.sin(pan_angle) #pans to side based on how close camera is to EH
##     camera_dir = ti.Vector(new_dir.tolist())
##
##
##def generate_video():
##     global init_camera_pos,camera_dir
##     os.makedirs("frames", exist_ok=True) #create frames
##     
##     print("Initializing")
##     start_time = time.perf_counter()
##     frame = 0
##     
##     while frame < total_frames:
##          update_camera(init_t + (final_t - init_t) * frame / total_frames)
##          
##          inits(init_camera_pos,camera_dir,FOV)
##          render_rays()
##          ti.sync()
##          
##          name = f"frames/frame_{frame:04d}.png"
##          print_image(filename=name)
##          
##          if frame % 20 == 0:
##               print(f"Frame {frame} / {total_frames}")
##          frame += 1
##          
##          
##     end_time = time.perf_counter()
##     elapsed = end_time - start_time
##     print(f"Time: {elapsed:.4f}")
##     print('cd to OneDrive/Desktop/black/"mach 2"/frames')
##     print(f"Print in terminal: ffmpeg -framerate {FPS} -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
##
##generate_video()
##          
##          
     
     
     


