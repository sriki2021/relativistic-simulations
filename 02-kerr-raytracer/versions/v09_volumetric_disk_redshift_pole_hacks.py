import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" #idk why it was disabled

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import cv2

# Suppress warnings for invalid math in the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

#Changeables

dt = 0.001
radius = 50 #radius of world
a = 0.5     # 0 < a < 1, 1 is max, 0 is schwarzshield

#CAMERA VARS

width = 640
height = 360

##Resolutions (width x height):
##
##360p:   640 x 360
##720p:   1280 x 720
##1080p:  1920 x 1080
##1440p:  2560 x 1440
##4K UHD: 3840 x 2160 or 4096
##Beyond this can't be saved to memory

fov_x = 1.4
fov_y = 1.4

aliasing = True

accretion_disk = True
accretion_disk_r = 20
prograde = True

camera_dir =  np.array([0.0, -1.0, -0.05], dtype=np.float32)
init_camera_pos = np.array([0.0, 40.0, 1.0])






#Parameters
minx, maxx = -radius, radius
end = True
tau = 0.0
steps = 20000

pixel_width = (2.0 * fov_x) / width
pixel_height = (2.0 * (fov_y / (width/height))) / height

anti_alias_diff_x = pixel_width * 0.35 
anti_alias_diff_y = pixel_height * 0.35 

camera_pos = np.broadcast_to(init_camera_pos, (5, 3)) + np.array([[0.0, 0.0, 0.0],
                       [0.0,anti_alias_diff_x,0.0],
                       [0.0,-anti_alias_diff_x,0.0],
                       [0.0,0,0.0+anti_alias_diff_y],
                       [0.0,0,0.0-anti_alias_diff_y]], dtype=np.float32)

# Calxing functions
@njit
def outer_event_horizon_R(a):
     return 1 + np.sqrt(np.maximum(1 - a**2, 0.0))

@njit
def ISCO(a): #M = 1
     Z1 = 1 + ((1 - a**2) ** (1/3)) * ( np.power((1 + a),1/3) + np.power((1 - a),1/3) )
     Z2 = np.sqrt(3*a**2 + Z1**2)

     r_ISCO = 3 + Z2 - np.sqrt( (3 - Z1) * (3 + Z1 + 2*Z2) )
     return r_ISCO

def potential(x, y, a):
    r = np.sqrt(x**2 + y**2)
    v = -r/(r**2 + a**2)
    return v + 1 #(Constant to fit the graph)

@njit
def to_spherical(pos):
    w2 = pos[0]**2 + pos[1]**2 + pos[2]**2
    r = np.sqrt(np.abs(w2 - a**2 + np.sqrt(np.abs((w2 - a**2)**2 + 4 * a**2 * pos[2]**2))) / 2)
    theta = np.arccos(min(max(pos[2] / r, -1.0), 1.0)) #clip to avoid errors
    phi = np.arctan2(pos[1],pos[0])
    return r,theta,phi

@njit
def to_cartesian(pos):
    r = pos[0]
    theta = pos[1]
    phi = pos[2]
    
    x = np.sqrt(np.maximum(r**2+a**2,0.0)) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(np.maximum(r**2+a**2,0.0)) * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z

#delete these functions later
@njit
def latitudinal_momentum(v, r, theta, phi):
    p_theta = (v[0] * np.cos(theta) * np.cos(phi)) + (v[1] * np.cos(theta) * np.sin(phi)) + (-v[2] * np.sin(theta))
    return p_theta * r

@njit
def Carter_Constant(L_z, a, v, r, theta, phi):
    L_theta = latitudinal_momentum(v, r, theta, phi)
    Q = L_theta**2 + np.cos(theta)**2 * (L_z**2 / (np.sin(theta)**2 + 1e-10) - a**2)
    return Q

@njit
def delta(r,a):
    return r**2 + a**2 - 2*r

@njit
def axial_ang_momentum(pos, v): #specific ang momentum
    mod_v = np.sqrt(v[0]**2 + v[1]**2)
    return (pos[0]*v[1] - pos[1]*v[0])/mod_v if mod_v != 0 else 0

@njit()
def calculate_derivatives(r, theta, phi, pr, ptheta, a, L_z, Q):
     #Suspicious convention where pr and ptheta represent derivatives of p and theta (w.r.t mino time)
     #but only the terms d_r and d_theta are used to calculate r and theta (so that they can update independantly)
     #d_pr and d_ptheta represent second derivatives of r and theta

     #phi doesn't need 2nd derivatives as phi doesnt depend on previous d_phi

     secure_theta = max(min(theta, np.pi - 1e-7), 1e-7) #avoid singularities at theta = 0 and pi - didn't really work
     sin_theta = np.sin(secure_theta)
     cos_theta = np.cos(secure_theta)
     
     d_pr = 2*r*(r**2 + a**2 - a*L_z) - (r - 1)*(Q + (L_z - a)**2)
     d_ptheta = cos_theta * (L_z**2 / (sin_theta**3) - a**2 * sin_theta)

     d_r = pr
     d_theta = ptheta
     d_phi = a / (delta(r,a) + 1e-10) * ((r**2 + a**2) - a*L_z) - a + L_z / (sin_theta**2)

     return d_r, d_theta, d_phi, d_pr, d_ptheta

@njit()
def calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, step): # RK4

    near_pole = (theta < 1e-3) or (theta > np.pi - 1e-3) #more SAFETY cuz boyer-lindquist fail at pole

    d_r1, d_theta1, d_phi1, d_pr1, d_ptheta1 = calculate_derivatives(r, theta, phi, pr, ptheta, a, L_z, Q)
    if near_pole: d_phi1 = 0 
    
    r2 = r + d_r1 * step/2
    theta2 = theta + d_theta1 * step/2
    phi2 = phi + d_phi1 * step/2
    pr2 = pr + d_pr1 * step/2
    ptheta2 = ptheta + d_ptheta1 * step/2

    d_r2, d_theta2, d_phi2, d_pr2, d_ptheta2 = calculate_derivatives(r2, theta2, phi2, pr2, ptheta2, a, L_z, Q)
    if near_pole: d_phi2 = 0 

    r3 = r + d_r2 * step/2
    theta3 = theta + d_theta2 * step/2
    phi3 = phi + d_phi2 * step/2
    pr3 = pr + d_pr2 * step/2
    ptheta3 = ptheta + d_ptheta2 * step/2

    d_r3, d_theta3, d_phi3, d_pr3, d_ptheta3 = calculate_derivatives(r3, theta3, phi3, pr3, ptheta3, a, L_z, Q)
    if near_pole: d_phi3 = 0
     
    r4 = r + d_r3 * step
    theta4 = theta + d_theta3 * step
    phi4 = phi + d_phi3 * step
    pr4 = pr + d_pr3 * step
    ptheta4 = ptheta + d_ptheta3 * step

    d_r4, d_theta4, d_phi4, d_pr4, d_ptheta4 = calculate_derivatives(r4, theta4, phi4, pr4, ptheta4, a, L_z, Q)
    if near_pole: d_phi4 = 0

    r_final = r + step/6 * (d_r1 + 2 * d_r2 + 2 * d_r3 + d_r4)
    theta_final = theta + step/6 * (d_theta1 + 2 * d_theta2 + 2 * d_theta3 + d_theta4)
    pr_final = pr + step/6 * (d_pr1 + 2 * d_pr2 + 2 * d_pr3 + d_pr4)
    ptheta_final = ptheta + step/6 * (d_ptheta1 + 2 * d_ptheta2 + 2 * d_ptheta3 + d_ptheta4)
    phi_final = phi + step/6 * (d_phi1 + 2 * d_phi2 + 2 * d_phi3 + d_phi4)

    if theta_final < 0.0:
        phi_final = np.pi + phi
        theta_final = - theta_final
        ptheta_final = - ptheta_final
    elif theta_final > np.pi:
        phi_final = np.pi + phi
        theta_final = 2*np.pi - theta_final
        ptheta_final = - ptheta_final
        
    
    
    return r_final,theta_final,phi_final,pr_final,ptheta_final

@njit
def update(pos,pvelo, inward, L_z, a, Q, dt, r_ISCO, transmittance, intensity):
    r = pos[0]
    theta = pos[1]
    phi = pos[2]
    
    pr = pvelo[0]
    ptheta = pvelo[1]
    
    r_new,theta_new,phi_new,pr_new,ptheta_new = calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, dt)
    pos_new = np.array([r_new,theta_new,phi_new])
    pvelo_new = np.array([pr_new,ptheta_new])
    
    state_new,trans_new,intens_new = checkCollided(pos,pos_new,pvelo,pvelo_new,L_z,a,r_ISCO, transmittance, intensity)
    return pos_new,inward, pvelo_new, state_new,trans_new,intens_new

#Some components of the metric needed below:
@njit
def g_tt(r):
     return -1.0 * (1 - 2 / r)
@njit
def g_tphi(r,a):
     return -2.0 * a / r
@njit
def g_phiphi(r,a):
     return (r ** 2) + (a ** 2) + 2 * (a ** 2) / r


#@njit
#def Q(




#B and C are relativistic corrections for frame dragging near the event horizon
@njit
def B(x,a):                        
     return 1.0 + a * (x ** -3)

@njit
def C(x,a):
     return 1 - 3 * (x ** -2) + 2 * a * (x ** -3)

#Accounts for kinematic doppler shift and gravitational redshift
@njit
def redshift_factor(x,a,v,r,b,theta,phi):
     sign = 1.0 if prograde else -1.0
     
     omega = 1 / (x ** 3 + sign*a) #relativistic angular velocity
     numerator = np.sqrt( -g_tt(r) - 2 * omega * g_tphi(r,a) - (omega ** 2) * g_phiphi(r,a))# time dilation
     
     g = numerator / ((1 - omega*b))
     return g

@njit
def vertical_density(r,z,r_ISCO):
     n0 = 1 # Baseline optical thickness at inner edge
     H0 = 0.2 # Scale height factor - increasing makes it more puffy
     p = 1.125 # Flaring Power index - determines flaring at upper/lower edges of disk
     n_mid = n0 * (r / r_ISCO) ** -1.5
     H = H0 * r_ISCO * (r / r_ISCO) ** p #gaussian vertical profile
     density  = n_mid * np.exp(-z**2/(2*H**2))
     return density


@njit
def radial_temperature(r, r_ISCO):
     if r < r_ISCO: return 0.0
     M_dot = 3 # Mass accretion rate - determines the glow linearly
     T = (M_dot/(r**3) * (1 - np.sqrt(r_ISCO/r))) ** 0.25 ## TO FIX - newtonian approximation 
     return T
     
@njit
def source(T):
     C = 2000.0 # Camera exposure slider
     S = C * T**4
     return S


@njit
def Volumetric_Calculations(r, theta, phi, v, z, L_z, transmittance, intensity, r_ISCO, delta_s):
    # Using the Novikov-Thorne Model
    x = np.sqrt(r)

    aa = 0.1 # Absorption coefficient - Control how opaque the disk is
    t = aa * vertical_density(r,z,r_ISCO) * delta_s #opacity
    T = np.exp(-t) #Transmission

    Temp = radial_temperature(r, r_ISCO)
    S_i = source(Temp)
    E_i = S_i * (1 - T)

    I_new = intensity + (E_i * transmittance) * (redshift_factor(x,a,v,r,L_z,theta,phi) ** 4)
    Transm_new = transmittance * T
    state = 0
    if Transm_new < 1e-5:
         state = 2
    return Transm_new, I_new, state

##-> opacity t = a * n(r,z) * delta_s
##-> Transmission T = exp(-t) // 0 completely opaque, 1 transparent
##
##-> emissitivty j -> calx normalized temperature T(r) convert to RGB colour
##-> Emission: E = (j(T) / a) * (1 - delta_s)
##
##calx intensity I' = I + (E * Tcurrent)       ## not I' = (I * T(r)) + E(r)
##Transmittance Tcurrent = Tcurrent * T
##
##if Tcurrent < 0.001: early termination
##if escaper to infinite: Ifinal = I + (Isky * Tcurrent)



@njit
def checkCollided(prev_pos, pos,prev_velo,velo,L_z, a, r_ISCO, transmittance, intensity):
    disk_thickness=0.8
    prev_x,prev_y,prev_z = to_cartesian(prev_pos)
    x,y,z = to_cartesian(pos)
    r = pos[0]
    theta = pos[1]
    phi = pos[2]

    horizon = outer_event_horizon_R(a) + 0.02

    if r <= horizon:
         return 1,transmittance, intensity

    inside_disk = (np.abs(z) <= disk_thickness / 2.0) and (r > r_ISCO) and (r < r_ISCO+accretion_disk_r)
    if accretion_disk and inside_disk : #Kinda arbitrary, just testing values for now

         delta_s = np.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)

         transmittance, intensity, state = Volumetric_Calculations(r, theta, phi, velo, z, L_z, transmittance, intensity, r_ISCO, delta_s)
         return state,transmittance, intensity
    if (r > maxx):
         return 3,transmittance, intensity
    return 0,transmittance, intensity


#plotter = pv.Plotter()
#camera

ratio = width / height
screen = (-1.0*fov_x, 1.0 * fov_y / ratio, 1.0*fov_x, -1.0 * fov_y / ratio) # left, top, right, bottom
image = np.zeros((height, width, 1))

tau = 0
#number of photons
num = width * height

#vector functions
@njit
def norm(v):
     normalized = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
     return v / normalized if normalized > 0 else v
@njit
def cross_product(v1, v2):
     cross = np.array([ v1[1]*v2[2] - v1[2]*v2[1] , v1[2]*v2[0] - v1[0]*v2[2] , v1[0]*v2[1] - v1[1]*v2[0] ], dtype=np.float32)
     return cross
@njit
def dot_product(v1, v2):
     dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
     return dot

@njit(parallel=True)
def inits(pos, velo, pvelo, inward, a, L_z, Q, camera_pos, camera_dir):


    # camera vectors fun
     
    #obtain the 3 unit vectors relative to camera pos: forward up and right
    forward = norm(camera_dir)
    fake_up = np.array([0,0,1], dtype=np.float32) #used to obtain vector perp to forward
    right = norm(cross_product(forward,fake_up))
    up = norm(cross_product(right, forward)) #actual up vector
    
    step1 = (screen[2]-screen[0]) / (width-1)
    step2 = (screen[3]-screen[1]) / (height-1)

    for j in range(len(pos)):
        for i in prange(len(pos[0])):
            pos[j,i] = camera_pos[j]

    for j in range(len(pos)):
         for i in prange(height):
             for k in range(width):
                 index = width*(i) + k
                 sx = screen[0] + k*step1
                 sy = screen[1] + i*step2

                 ray_dir = forward + sx * right + sy * up # Combine the vectors
                 velo[j,index] = norm(ray_dir)
        
         for i in prange(len(pos[0])):

              p = pos[j,i]
              v = velo[j,i]

              r,theta,phi = to_spherical(p)

              #Unit vectors for calxing
              e_r = np.array([
                  np.sin(theta)*np.cos(phi),           #NOTE: This is regular spherical not Boyer Lindquist so camera should kinda far from horizon
                  np.sin(theta)*np.sin(phi), 
                  np.cos(theta)
              ], dtype=np.float32)
              
              e_theta = np.array([
                  np.cos(theta)*np.cos(phi), 
                  np.cos(theta)*np.sin(phi), 
                  -np.sin(theta)
              ], dtype=np.float32)

              v_r = dot_product(p,v)
              v_theta = dot_product(v,e_theta)

              inward[j,i,0] = v_r < 0
              inward[j,i,1] = v_theta < 0
                           
              L_z_init = p[0]*v[1] - p[1]*v[0]
              p_theta_init = v_theta * r
              Q_init = p_theta_init**2 + np.cos(theta)**2 * (L_z_init**2 / (np.sin(theta)**2 + 1e-10) - a**2) #no more functions
                            
              L_z[j,i] = L_z_init
              Q[j,i] = Q_init
              
              rad_v = np.sqrt(np.maximum(((r**2 + a**2) - a*L_z_init)**2 - delta(r,a)*(Q_init + (L_z_init - a)**2), 0.0))
              pvelo[j,i,0] = -rad_v if inward[j,i,0] else rad_v
              
              cot_sq = (np.cos(theta) / (np.sin(theta) + 1e-10))**2 #avoid division by 0
              
              theta_v = np.sqrt(np.maximum(Q_init + a**2 * np.cos(theta)**2 - L_z_init**2 * cot_sq, 0.0))
              pvelo[j,i,1] = -theta_v if inward[j,i,1] else theta_v

              pos[j,i,0] = r
              pos[j,i,1] = theta
              pos[j,i,2] = phi

    return pos, velo, pvelo, inward, L_z, Q



pos = np.zeros((5,num,3),dtype=np.float32)
velo = np.zeros((5,num,3))
pvelo = np.zeros((5,num,2),dtype=np.float32)
L_z = np.zeros((5,num,),dtype=np.float32)
Q = np.zeros((5,num,),dtype=np.float32)
inward = np.zeros((5,num,2),dtype=bool) #For 'r' and 'theta' 
state = np.zeros((5,num,))
transmittance = np.ones((5,num,),dtype=np.float32)
intensity = np.zeros((5,num,),dtype=np.float32)
r_ISCO = ISCO(a)

@njit(parallel=True)
def updateParticles(pos, pvelo, L_z, state, inward, Q, r_ISCO, transmittance, intensity):
     for j in range(len(pos)):
         for i in prange(len(pos[0])):
             if state[j,i] == 0:
                 r_prev = pos[j,i,0]
                 theta_prev = pos[j,i,1]
                 phi_prev = pos[j,i,2]
                 pos_new,inward_new, pvelo_new, state_new,trans_new,intens_new = update(pos[j,i], pvelo[j,i], inward[j,i], L_z[j,i], a, Q[j,i], dt, r_ISCO,  transmittance[j,i], intensity[j,i])

                 if state_new == 3: #interpolation for smoothing  
                      r_new = pos_new[0]
                      fraction = (maxx - r_prev) / (r_new - r_prev + 1e-10)

                      pos[j,i,0] = maxx
                      pos[j,i,1] = theta_prev + fraction * (pos_new[1] - theta_prev)
                      pos[j,i,2] = phi_prev + fraction * (pos_new[2] - phi_prev)
                 elif state_new == 2:
                      z_prev = r_prev * np.cos(theta_prev)
                      z_new = pos_new[0] * np.cos(pos_new[1])
                      fraction = (0 - z_prev) / (z_new - z_prev + 1e-10)

                      pos[j,i,0] = r_prev + fraction * (pos_new[0] - r_prev)
                      pos[j,i,1] = np.pi / 2.0 
                      pos[j,i,2] = phi_prev + fraction * (pos_new[2] - phi_prev)
                 else:
                      pos[j,i] = pos_new
                 inward[j,i] = inward_new
                 pvelo[j,i] = pvelo_new
                 state[j,i] = state_new
                 transmittance[j,i] = trans_new
                 intensity[j,i] = intens_new

def load_hdri(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load HDRI file at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

@njit
def sample_hdri(theta,phi, hdri_map):
    theta = theta % (2*np.pi) #safety first
    if theta > np.pi:
        theta = 2 * np.pi - theta
        phi += np.pi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    
    u = 0.5 + phi / (2 * np.pi)
    v = theta / np.pi

    height = hdri_map.shape[0]
    width = hdri_map.shape[1] 
    x = int(u * (width - 1)) % width
    y = int(v * (height - 1)) % height

    return hdri_map[y, x]


hdri_map = load_hdri("starmap_4k.exr")

##@njit
##def calc_temperature(r, r_ISCO):
##     if r <= r_ISCO:
##          return np.array([0,0,0],dtype=np.float32)
##
##     #Using the Shakura-Sunyaev thin accretion disk model for temperatire
##     
##     temp_profile = ((r_ISCO / r) ** 0.75) * ((1.0 - np.sqrt(r_ISCO/r)) ** 0.25)
##
##     intensity = temp_profile * 2.1
##     intensity = intensity**3.0
##
##     #fade_start = 15 #start fading the brightness after this
##     
##     #if 
##
##
##     intensity = min(max(intensity, 0.0), 1.0)
##
##
##     return np.array([intensity,intensity,intensity], dtype=np.float32)
     

##     #temp_kelvin = 1000.0 + (19000.0 * temp_profile / 0.1) #making range of values
##
##     norm_t = (temp_profile - 0.28) / (0.48 - 0.28) #dodgification
##     norm_t = max(min(norm_t, 1.0), 0.0)
##     temp_kelvin = 1000.0 + (norm_t * 29000.0)
##
##     #Tanner Helland's algorithm to convert Thermodynamic temperature into RGB
##     
##     temp = min(max(temp_kelvin / 100.0, 10.0), 400.0)
##
##     if temp <= 66.0:
##        red = 255.0
##     else:
##        red = temp - 60.0
##        red = 329.698727446 * (red ** -0.1332047592)
##        red = min(max(red, 0.0), 255.0)
##        
##     # Green
##     if temp <= 66.0:
##        green = temp
##        green = 99.4708025861 * np.log(green) - 161.1195681661
##     else:
##        green = temp - 60.0
##        green = 288.1221695283 * (green ** -0.0755148492)
##     green = min(max(green, 0.0), 255.0)
##        
##     # Blue
##     if temp >= 66.0:
##        blue = 255.0
##     elif temp <= 19.0:
##        blue = 0.0
##     else:
##        blue = temp - 10.0
##        blue = 138.5177312231 * np.log(blue) - 305.0447927307
##     blue = min(max(blue, 0.0), 255.0)
##
##     intensity = min(temp_profile * 15.0, 1.0) #gets dimmer further away

##     return np.array([red * intensity / 255.0, green * intensity / 255.0, blue * intensity / 255.0], dtype=np.float32)

##if escaper to infinite: Ifinal = I + (Isky * Tcurrent)

@njit
def render_colours(state, pos, r_ISCO,transmittance, intensity, hdri_map):
     colours = np.zeros((5,num,3),dtype=np.float32) #default is landing in event horizon
     for i in range(len(state)):
          for j in prange(len(state[0])):
               # 1 - Horizon, 2 - Accretion Disk, 3 - Escaper
               s = state[i,j]
               
               r = 0.0
               g = 0.0
               b = 0.0
               if s == 3:
                    theta = pos[i,j,1]
                    phi = pos[i,j,2]
                    c_sky = sample_hdri(theta,phi, hdri_map)
                    r = c_sky[0] * transmittance[i,j]
                    g = c_sky[1] * transmittance[i,j]
                    b = c_sky[2] * transmittance[i,j]

               disk_intensity = intensity[i,j]
               r += disk_intensity
               g += disk_intensity
               b += disk_intensity

               #Reinhard Tone mapping - compresses light pixels but leaves dark pixels alone for the screen
               r_map = r / (r + 1.0)
               g_map = g / (g + 1.0)
               b_map = b / (b + 1.0)
               
               #gamma correction - makes image look more natural to human eye when displayed
               colours[i,j,0] = np.power(r_map, 1/2.2)
               colours[i,j,1] = np.power(g_map, 1/2.2)
               colours[i,j,2] = np.power(b_map, 1/2.2)

     return colours
          

def print_image(state, pos, r_ISCO,transmittance, intensity, hdri_map):
     colours = render_colours(state, pos, r_ISCO,transmittance, intensity, hdri_map)
     avg_colours = np.zeros((num,3))
     

     #if aliasing:
     weight_centre = 0.5
     weight_sides = 0.125
     for i in range(len(state)):
          if i == 0:
               avg_colours += colours[i] * weight_centre
          else:
               avg_colours += colours[i] * weight_sides
     #else:
     #     avg_colours = colours[0]
     
     image = np.reshape(avg_colours, (height, width, 3))
     image = np.clip(image, 0.0, 1.0)
     plt.imsave('balck.png', image)
     print("done")

@njit
def main_loop(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO,  transmittance, intensity):
     for i in range(steps):
        if not np.any(state == 0):
             break
        updateParticles(pos, pvelo, L_z, state, inward, Q, r_ISCO, transmittance, intensity)
        tau += dt

def simulate_photons(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO,transmittance, intensity, hdri_map):
    print("helo")
    tau = 0
    start_time = time.perf_counter()
    
    pos, velo, pvelo, inward, L_z, Q = inits(pos, velo, pvelo, inward, a, L_z, Q, camera_pos, camera_dir)
    main_loop(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO,transmittance, intensity)

    print("donzo")
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f}, Resolution: {width} x {height}, dt: {dt}")
    print_image(state, pos, r_ISCO,transmittance, intensity, hdri_map)
    

simulate_photons(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO, transmittance, intensity, hdri_map)



    
