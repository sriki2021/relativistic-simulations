import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" #idk why it was disabled

import numpy as np
import taichi as ti
import time
import cv2

ti.init(arch=ti.cuda) #using nvidia gpu

# Suppress warnings for invalid math in the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

# best render options: a = 0, tol = 7-8, 4K res, 

#Changeables

radius = 50 #radius of world
a = 0.5     # 0 < a < 1, 1 is max, 0 is schwarzshield
dt_start = 0.001
dt_min = 1e-10 #e-10 orig -8 fine
dt_max = 0.1
tol = 1e-6 # 1e-8: 60s, Stunning (no tiny artefacts whatsoever), 1e-6: 2s best choice

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

X_FOV = 110
Y_FOV = 110

aliasing = True

accretion_disk = True
accretion_disk_r = 20
prograde = False #doesn't do anything

init_camera_pos = ti.Vector([0.0, 15.0, 2.0]) #orig [0,40,1]
camera_dir =  -1*init_camera_pos #ti.Vector([0.0, -1.0, -0.05])








#Parameters
steps = 20000

fov_x = np.tan(np.deg2rad(X_FOV)/2)
fov_y = np.tan(np.deg2rad(Y_FOV)/2)
pixel_width = (2.0 * fov_x) / width
pixel_height = (2.0 * (fov_y / (width/height))) / height

diff_x = pixel_width * 0.35 
diff_y = pixel_height * 0.35 

offsets = ti.Vector.field(n=3, dtype=ti.f32, shape=5)
#offsets = ti.field([[0.0, 0.0, 0.0],[0.0,anti_alias_diff_x,0.0],[0.0,-anti_alias_diff_x,0.0],[0.0,0,0.0+anti_alias_diff_y],[0.0,0,0.0-anti_alias_diff_y]], dtype=ti.f32)
camera_pos = ti.Vector.field(n=3,dtype=ti.f32,shape=(5,))


num_rays = 5 * width * height #num photons

pos   = ti.Vector.field(3, dtype=ti.f32, shape=(5,height,width))
velo  = ti.Vector.field(3, dtype=ti.f32, shape=(5,height,width))
pvelo = ti.Vector.field(2, dtype=ti.f32, shape=(5,height,width))

L_z           = ti.field(dtype=ti.f32,shape=(5,height,width))
Q             = ti.field(dtype=ti.f32,shape=(5,height,width))
state         = ti.field(dtype=ti.i32,shape=(5,height,width))
dt            = ti.field(dtype=ti.f32,shape=(5,height,width))
transmittance = ti.field(dtype=ti.f32,shape=(5,height,width))
intensity     = ti.field(dtype=ti.f32,shape=(5,height,width))

inward = ti.Vector.field(2, dtype=ti.i32, shape=(5,height,width)) #For 'r' and 'theta'

colours = ti.Vector.field(n=3,shape=(5,height,width),dtype=ti.f32)
avg_colours = ti.Vector.field(n=3,shape=(height,width),dtype=ti.f32)

# Calxing functions
@ti.func
def outer_event_horizon_R(a):
     return 1 + ti.math.sqrt(ti.max(1 - a**2, 0.0))

def ISCO(a): #M = 1
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

###delete these functions later
##@ti.func
##def latitudinal_momentum(v, r, theta, phi):
##    p_theta = (v[0] * ti.math.cos(theta) * ti.math.cos(phi)) + (v[1] * ti.math.cos(theta) * ti.math.sin(phi)) + (-v[2] * ti.math.sin(theta))
##    return p_theta * r
##
##@ti.func
##def Carter_Constant(L_z, a, v, r, theta, phi):
##    L_theta = latitudinal_momentum(v, r, theta, phi)
##    Q = L_theta**2 + ti.math.cos(theta)**2 * (L_z**2 / (ti.math.sin(theta)**2 + 1e-10) - a**2)
##    return Q

@ti.func
def delta(r,a):
    return r**2 + a**2 - 2*r

##@ti.func
##def axial_ang_momentum(pos, v): #specific ang momentum
##    mod_v = ti.math.sqrt(v[0]**2 + v[1]**2)
##    return (pos[0]*v[1] - pos[1]*v[0])/mod_v if mod_v != 0 else 0

@ti.func
def calculate_derivatives(r, theta, phi, pr, ptheta, a, L_z, Q):
     #Suspicious convention where pr and ptheta represent derivatives of p and theta (w.r.t mino time)
     #but only the terms d_r and d_theta are used to calculate r and theta (so that they can update independantly)
     #d_pr and d_ptheta represent second derivatives of r and theta

     #phi doesn't need 2nd derivatives as phi doesnt depend on previous d_phi

     #secure_theta = max(min(theta, ti.math.pi - 1e-7), 1e-7) #avoid singularities at theta = 0 and pi - didn't really work
     sin_theta = ti.math.sin(theta)
     cos_theta = ti.math.cos(theta)
     
     d_pr = 2*r*(r**2 + a**2 - a*L_z) - (r - 1)*(Q + (L_z - a)**2)
     d_ptheta = cos_theta * (L_z**2 / (sin_theta**3) - a**2 * sin_theta)

     d_r = pr
     d_theta = ptheta
     d_phi = a / (delta(r,a) + 1e-10) * ((r**2 + a**2) - a*L_z) - a + L_z / (sin_theta**2)

     return d_r, d_theta, d_phi, d_pr, d_ptheta

# NEW dormand prince RK-45 method :O
@ti.func
def calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, step):
     shrunked = False
     step_new = step
     r7 = theta7 = phi7 = pr7 = ptheta7 = 0.0
     if ti.abs(theta - ti.math.pi/2) < 0.1:
          step = max(1e-5,0.1*ti.abs(theta - ti.math.pi/2))
          shrunked = True
     
     for i in range(50): #max count so no infinite loops
         d_r1, d_theta1, d_phi1, d_pr1, d_ptheta1 = calculate_derivatives(r, theta, phi, pr, ptheta, a, L_z, Q) #k1
         
         r2 = r + d_r1 * step/5
         theta2 = theta + d_theta1 * step/5
         phi2 = phi + d_phi1 * step/5
         pr2 = pr + d_pr1 * step/5
         ptheta2 = ptheta + d_ptheta1 * step/5

         d_r2, d_theta2, d_phi2, d_pr2, d_ptheta2 = calculate_derivatives(r2, theta2, phi2, pr2, ptheta2, a, L_z, Q)

         r3 = r + (3*d_r1 + 9*d_r2) * step/40
         theta3 = theta + (3*d_theta1 + 9*d_theta2) * step/40
         phi3 = phi + (3*d_phi1 + 9*d_phi2) * step/40
         pr3 = pr + (3*d_pr1 + 9*d_pr2) * step/40
         ptheta3 = ptheta + (3*d_ptheta1 + 9*d_ptheta2) * step/40

         d_r3, d_theta3, d_phi3, d_pr3, d_ptheta3 = calculate_derivatives(r3, theta3, phi3, pr3, ptheta3, a, L_z, Q)
          
         r4 = r + (44/45 * d_r1 - 56/15 * d_r2 + 32/9*d_r3) * step
         theta4 = theta + (44/45 * d_theta1 - 56/15 * d_theta2 + 32/9*d_theta3) * step
         phi4 = phi + (44/45 * d_phi1 - 56/15 * d_phi2 + 32/9*d_phi3) * step
         pr4 = pr + (44/45 * d_pr1 - 56/15 * d_pr2 + 32/9*d_pr3) * step
         ptheta4 = ptheta + (44/45 * d_ptheta1 - 56/15 * d_ptheta2 + 32/9*d_ptheta3) * step

         d_r4, d_theta4, d_phi4, d_pr4, d_ptheta4 = calculate_derivatives(r4, theta4, phi4, pr4, ptheta4, a, L_z, Q)

         r5 = r + (19372/6561 * d_r1 - 25360/2187 * d_r2 + 64448/6561 * d_r3 - 212/729 * d_r4) * step
         theta5 = theta + (19372/6561 * d_theta1 - 25360/2187 * d_theta2 + 64448/6561 * d_theta3 - 212/729 * d_theta4) * step
         phi5 = phi + (19372/6561 * d_phi1 - 25360/2187 * d_phi2 + 64448/6561 * d_phi3 - 212/729 * d_phi4) * step
         pr5 = pr + (19372/6561 * d_pr1 - 25360/2187 * d_pr2 + 64448/6561 * d_pr3 - 212/729 * d_pr4) * step
         ptheta5 = ptheta + (19372/6561 * d_ptheta1 - 25360/2187 * d_ptheta2 + 64448/6561 * d_ptheta3 - 212/729 * d_ptheta4) * step

         d_r5, d_theta5, d_phi5, d_pr5, d_ptheta5 = calculate_derivatives(r5, theta5, phi5, pr5, ptheta5, a, L_z, Q)

         r6 = r + (9017/3168 * d_r1 - 355/33 * d_r2 + 46732/5247 * d_r3 + 49/176 * d_r4 - 5103/18656 * d_r5) * step
         theta6 = theta + (9017/3168 * d_theta1 - 355/33 * d_theta2 + 46732/5247 * d_theta3 + 49/176 * d_theta4 - 5103/18656 * d_theta5) * step
         phi6 = phi + (9017/3168 * d_phi1 - 355/33 * d_phi2 + 46732/5247 * d_phi3 + 49/176 * d_phi4 - 5103/18656 * d_phi5) * step
         pr6 = pr + (9017/3168 * d_pr1 - 355/33 * d_pr2 + 46732/5247 * d_pr3 + 49/176 * d_pr4 - 5103/18656 * d_pr5) * step
         ptheta6 = ptheta + (9017/3168 * d_ptheta1 - 355/33 * d_ptheta2 + 46732/5247 * d_ptheta3 + 49/176 * d_ptheta4 - 5103/18656 * d_ptheta5) * step

         d_r6, d_theta6, d_phi6, d_pr6, d_ptheta6 = calculate_derivatives(r6, theta6, phi6, pr6, ptheta6, a, L_z, Q)


         #5th order solutions
         r7 = r + (35/384 * d_r1 + 500/1113 * d_r3 + 125/192 * d_r4 - 2187/6784 * d_r5 + 11/84 * d_r6) * step
         theta7 = theta + (35/384 * d_theta1 + 500/1113 * d_theta3 + 125/192 * d_theta4 - 2187/6784 * d_theta5 + 11/84 * d_theta6) * step
         phi7 = phi + (35/384 * d_phi1 + 500/1113 * d_phi3 + 125/192 * d_phi4 - 2187/6784 * d_phi5 + 11/84 * d_phi6) * step
         pr7 = pr + (35/384 * d_pr1 + 500/1113 * d_pr3 + 125/192 * d_pr4 - 2187/6784 * d_pr5 + 11/84 * d_pr6) * step
         ptheta7 = ptheta + (35/384 * d_ptheta1 + 500/1113 * d_ptheta3 + 125/192 * d_ptheta4 - 2187/6784 * d_ptheta5 + 11/84 * d_ptheta6) * step

         d_r7, d_theta7, d_phi7, d_pr7, d_ptheta7 = calculate_derivatives(r7, theta7, phi7, pr7, ptheta7, a, L_z, Q)

         #rather than calxing 4th and subtracting, direct error computed:
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
              if shrunked:
                   step_new = max(step_new,0.0)
              else:
                   step_new = max(step / 5.0, min(step * 5.0, step_new))     
              step_new = max(dt_min,min(dt_max,step_new))
              
              #return r7,theta7,phi7,pr7,ptheta7,step_new
              break
         else:
              step = 0.9 * step * (tol / error) ** 0.25
              if step < dt_min:
                  #r7 = 0.0
                  step_new = dt_min
                  break
                   #return r7,theta7,phi7,pr7,ptheta7,dt_min

     return r7,theta7,phi7,pr7,ptheta7,step_new
          
         ### Consider passing dr_7 etc so that it can be reused
            
    

@ti.func
def update(pos,pvelo, inward, L_z, a, Q, dt, r_ISCO, transmittance, intensity):
    r = pos[0]
    theta = pos[1]
    phi = pos[2]
    
    pr = pvelo[0]
    ptheta = pvelo[1]
    
    r_new,theta_new,phi_new,pr_new,ptheta_new,dt_new = calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, dt)
    pos_new = ti.Vector([r_new,theta_new,phi_new])
    pvelo_new = ti.Vector([pr_new,ptheta_new])
    
    state_new,trans_new,intens_new = checkCollided(pos,pos_new,pvelo,pvelo_new,L_z,a,r_ISCO, transmittance, intensity)
    return pos_new,inward, pvelo_new, state_new,dt_new,trans_new,intens_new

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

          sum_term += (3*(xn[i] - a)**2 * ti.math.log((x - xn[i])/(x_ISCO - xn[i]))) / (xn[i] * B_val * (xn[i] - xn[j]) * (xn[i] - xn[k]))
          
     return term_1 - sum_term




#B and C are relativistic corrections for frame dragging near the event horizon
@ti.func
def B_f(x,a):                        
     return 1.0 + a * (x ** -3)

@ti.func
def C_f(x,a):
     return 1 - 3 * (x ** -2) + 2 * a * (x ** -3)

#Accounts for kinematic doppler shift and gravitational redshift
@ti.func
def redshift_factor(x,a,v,r,b,theta,phi):
     sign = 1.0 if prograde else -1.0
     
     omega = 1 / (x ** 3 + sign*a) #relativistic angular velocity
     numerator = ti.math.sqrt( -g_tt(r) - 2 * omega * g_tphi(r,a) - (omega ** 2) * g_phiphi(r,a))# time dilation from metric components
     
     g = numerator / ((1 - omega*b))
     return g

@ti.func
def vertical_density(r,z,r_ISCO):
     
     n0 = 1 # Baseline optical thickness at inner edge
     H0 = 0.2 # Scale height factor - increasing makes it more puffy
     p = 1.125 # Flaring Power index - determines flaring at upper/lower edges of disk
     fade_edge = 10 #disk starts fading
     n_mid = n0 * (r / r_ISCO) ** -1.5
     H = H0 * r_ISCO * (r / r_ISCO) ** p #gaussian vertical profile
     density  = n_mid * ti.math.exp(-z**2/(2*H**2))
     
     if r > accretion_disk_r - fade_edge:

          t = ti.max(0,ti.min(1,(accretion_disk_r-r)/fade_edge ))

          smooth_factor = t * t * (3.0 - 2.0 * t) #Using the smoothstep function to smoothen out the abrupt edge
          density *= smooth_factor
     
     return density


@ti.func
def radial_temperature(r, r_ISCO):
     M_dot = 3 # Mass accretion rate - determines the glow linearly
     T = 0.0
     if r >= r_ISCO - 1e-5:
         #new and improved (ugly) radial temperature
         r = ti.max(r,r_ISCO+1e-2)
         x = ti.math.sqrt(r)
         x_ISCO = ti.math.sqrt(r_ISCO)
         T = (M_dot/(r**3) * (Q_f(x,a,x_ISCO) / (B_f(x,a) * ti.math.sqrt(C_f(x,a))))) ** 0.25
         #T = (M_dot/(r**3) * (1 - ti.math.sqrt(r_ISCO/r))) ** 0.25 - oldie newtonian approximation
     return T
     
@ti.func
def source(T):
     C = 2000.0 # Camera exposure slider
     S = C * T**4
     return S


@ti.func
def Volumetric_Calculations(r, theta, phi, v, z, L_z, transmittance, intensity, r_ISCO, delta_s):
    Transm_new = transmittance
    I_new = intensity
    if transmittance > 1e-5:

         # Using the Novikov-Thorne Model
         x = ti.math.sqrt(r)
         aa = 0.1 # Absorption coefficient - Control how opaque the disk is
         t = aa * vertical_density(r,z,r_ISCO) * delta_s #opacity
         T = ti.math.exp(-t) #Transmission

         Temp = radial_temperature(r, r_ISCO)
         S_i = source(Temp)
         E_i = S_i * (1 - T)

         I_new = intensity + (E_i * transmittance) * (redshift_factor(x,a,v,r,L_z,theta,phi) ** 4)
         Transm_new = transmittance * T
    
    return Transm_new, I_new

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



@ti.func
def checkCollided(prev_pos, pos,prev_velo,velo,L_z, a, r_ISCO, transmittance, intensity):
    disk_thickness=0.8
    prev_x,prev_y,prev_z = to_cartesian(prev_pos)
    x,y,z = to_cartesian(pos)
    r = pos[0]
    theta = pos[1]
    phi = pos[2]

    horizon = outer_event_horizon_R(a) + 0.01

    new_state = 0
    if r <= horizon:
        new_state = 1

    inside_disk = (ti.abs(z) <= disk_thickness / 2.0) and (r > r_ISCO) and (r < r_ISCO+accretion_disk_r) #CHANGED
    if accretion_disk and inside_disk : #Kinda arbitrary, just testing values for now

         delta_s = ti.math.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)

         transmittance, intensity = Volumetric_Calculations(r, theta, phi, velo, z, L_z, transmittance, intensity, r_ISCO, delta_s)
    if (r > radius):
         new_state = 3
    return new_state,transmittance, intensity


#plotter = pv.Plotter()
#camera

ratio = width / height
screen = (-1.0*fov_x, 1.0 * fov_y / ratio, 1.0*fov_x, -1.0 * fov_y / ratio) # left, top, right, bottom
image = np.zeros((height, width, 1))

#vector functions
##@ti.func
##def norm(v):
##     normalized = ti.math.sqrt(v[0]**2+v[1]**2+v[2]**2)
##     return v / normalized if normalized > 0 else v
##@ti.func
##def cross_product(v1, v2):
##     cross = ti.Vector([ v1[1]*v2[2] - v1[2]*v2[1] , v1[2]*v2[0] - v1[0]*v2[2] , v1[0]*v2[1] - v1[1]*v2[0] ], dtype=ti.f32)
##     return cross
@ti.func
def dot_product(v1, v2):
     dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
     return dot

@ti.kernel
def inits():
    offsets[0] = ti.Vector([0.0, 0.0, 0.0])
    offsets[1] = ti.Vector([0.0, diff_x, 0.0])
    offsets[2] = ti.Vector([0.0, -diff_x, 0.0])
    offsets[3] = ti.Vector([0.0, 0.0, diff_y])
    offsets[4] = ti.Vector([0.0, 0.0, -diff_y])
    for i in range(5):
        camera_pos[i] = init_camera_pos + offsets[i]
     
    # camera vectors fun
    #obtain the 3 unit vectors relative to camera pos: forward up and right

    forward = ti.math.normalize(camera_dir)
    fake_up = ti.Vector([0.0,0.0,1.0]) #used to obtain vector perp to forward
    right = ti.math.normalize(ti.math.cross(forward,fake_up))
    up = ti.math.normalize(ti.math.cross(right, forward)) #actual up vector
    
    step1 = (screen[2]-screen[0]) / (width-1)
    step2 = (screen[3]-screen[1]) / (height-1)

    for j,i,k in ti.ndrange(5,height,width):
          c_pos = camera_pos[j]
          pos[j,i,k] = c_pos

          sx = screen[0] + k*step1
          sy = screen[1] + i*step2

          ray_dir = forward + sx * right + sy * up # Combine the vectors
          norm_dir = ti.math.normalize(ray_dir)
          velo[j,i,k] = norm_dir

          p = pos[j,i,k]
          v = velo[j,i,k]

          r,theta,phi = to_spherical(p)

          #Unit vectors for calxing
          e_r = ti.Vector([
             ti.math.sin(theta)*ti.math.cos(phi),           #NOTE: This is regular spherical not Boyer Lindquist so camera should kinda far from horizon
             ti.math.sin(theta)*ti.math.sin(phi), 
             ti.math.cos(theta)
          ])

          e_theta = ti.Vector([
             ti.math.cos(theta)*ti.math.cos(phi), 
             ti.math.cos(theta)*ti.math.sin(phi), 
             -ti.math.sin(theta)
          ])

          v_r = ti.math.dot(p,v)
          v_theta = ti.math.dot(v,e_theta)

          inward[j,i,k][0] = v_r < 0
          inward[j,i,k][1] = v_theta < 0
                      
          L_z_init = p[0]*v[1] - p[1]*v[0]
          p_theta_init = v_theta * r
          Q_init = p_theta_init**2 + ti.math.cos(theta)**2 * (L_z_init**2 / (ti.math.sin(theta)**2 + 1e-10) - a**2) #no more functions
                       
          L_z[j,i,k] = L_z_init
          Q[j,i,k] = Q_init

          rad_v = ti.math.sqrt(ti.max(((r**2 + a**2) - a*L_z_init)**2 - delta(r,a)*(Q_init + (L_z_init - a)**2), 0.0))
          pvelo[j,i,k][0] = -rad_v if inward[j,i,k][0] else rad_v

          cot_sq = (ti.math.cos(theta) / (ti.math.sin(theta) + 1e-10))**2 #avoid division by 0

          theta_v = ti.math.sqrt(ti.max(Q_init + a**2 * ti.math.cos(theta)**2 - L_z_init**2 * cot_sq, 0.0))
          pvelo[j,i,k][1] = -theta_v if inward[j,i,k][1] else theta_v

          pos[j,i,k][0] = r
          pos[j,i,k][1] = theta
          pos[j,i,k][2] = phi

          transmittance[j,i,k] = 1.0
          intensity[j,i,k] = 0.0

          dt[j,i,k] = dt_start

r_ISCO = ISCO(a)

@ti.kernel
def render_rays():
     for j,i,k in ti.ndrange(5,height,width):
          for step in range(steps):
              if state[j,i,k] != 0:
                 break

              r_prev = pos[j,i,k][0]
              theta_prev = pos[j,i,k][1]
              phi_prev = pos[j,i,k][2]
              pos_new,inward_new, pvelo_new, state_new,dt_new,trans_new,intens_new = update(pos[j,i,k], pvelo[j,i,k], inward[j,i,k], L_z[j,i,k], a, Q[j,i,k], dt[j,i,k], r_ISCO,  transmittance[j,i,k], intensity[j,i,k])

              if state_new == 3: #interpolation for smoothification
                   r_new = pos_new[0]
                   fraction = (radius - r_prev) / (r_new - r_prev + 1e-10)

                   pos[j,i,k][0] = radius
                   pos[j,i,k][1] = theta_prev + fraction * (pos_new[1] - theta_prev)
                   pos[j,i,k][2] = phi_prev + fraction * (pos_new[2] - phi_prev)
              elif state_new == 2:
                   z_prev = r_prev * ti.math.cos(theta_prev)
                   z_new = pos_new[0] * ti.math.cos(pos_new[1])
                   fraction = (0 - z_prev) / (z_new - z_prev + 1e-10)

                   pos[j,i,k][0] = r_prev + fraction * (pos_new[0] - r_prev)
                   pos[j,i,k][1] = ti.math.pi / 2.0 
                   pos[j,i,k][2] = phi_prev + fraction * (pos_new[2] - phi_prev)
              elif state_new == 1:
                  horizon = outer_event_horizon_R(a)
                  r_new = pos_new[0]
                  fraction = (horizon - r_prev) / (r_new - r_prev + 1e-10)
                  
                  pos[j,i,k][0] = horizon
                  pos[j,i,k][1] = theta_prev + fraction * (pos_new[1] - theta_prev)
                  pos[j,i,k][2] = phi_prev + fraction * (pos_new[2] - phi_prev)
                  

              else:
                   pos[j,i,k] = pos_new
              inward[j,i,k] = inward_new
              pvelo[j,i,k] = pvelo_new
              state[j,i,k] = state_new
              dt[j,i,k] = dt_new
              transmittance[j,i,k] = trans_new
              intensity[j,i,k] = intens_new

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
def sample_hdri(theta,phi):
    theta = theta % (2*ti.math.pi) #safety first
    if theta > ti.math.pi:
        theta = 2 * ti.math.pi - theta
        phi += ti.math.pi
    phi = (phi + ti.math.pi) % (2 * ti.math.pi) - ti.math.pi
    
    u = 0.5 + phi / (2 * ti.math.pi)
    v = theta / ti.math.pi

    height = hdri_map.shape[0]
    width = hdri_map.shape[1] 
    x = int(u * (width - 1)) % width
    y = int(v * (height - 1)) % height

    return hdri_map[y, x]

hdri_map = load_hdri("starmap_8k.exr")

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
                         theta = pos[i,j,k][1]
                         phi = pos[i,j,k][2]
                         c_sky = sample_hdri(theta,phi)
                         r = c_sky[0] * transmittance[i,j,k]
                         g = c_sky[1] * transmittance[i,j,k]
                         b = c_sky[2] * transmittance[i,j,k]

                    disk_intensity = intensity[i,j,k]
                    r += disk_intensity * 0.80
                    g += disk_intensity * 0.90
                    b += disk_intensity * 1.20 #subtle blue hint

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


def print_image():
    process_image()
    image = avg_colours.to_numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #cv2 uses bgr instead of rgb
    cv2.imwrite("balck.png", image_bgr)

def simulate_photons():
    print("helo")
    start_time = time.perf_counter()
    print("Initializing...")
    inits()
    print("Raytracing on GPU...")
    render_rays()
    ti.sync()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f}, Resolution: {width} x {height}, dt: {dt_start}")
    print_image()
    print("Render saved")
    
simulate_photons()
