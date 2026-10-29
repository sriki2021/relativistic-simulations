import pyvista as pv
import numpy as np
from numba import njit, prange
#import warnings
#import vtk
import matplotlib.pyplot as plt
import time

# Suppress warnings for invalid math in the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

#Parameters
minz, maxz = 0.90, 1.07
z_center = (maxz + minz) / 2
z_stretch = 10.0
tau = 0.0

dt = 0.001 #default accuracy
steps = 20000

radius = 20 #10 then 7

minx, maxx = -radius, radius 
end = True

#CAMERA VARS

width = 1280#640
height = 720#360

fov_x = 1.4
fov_y = 1.4

visuals = False


accretion_disk_r = 8

pixel_width = (2.0 * fov_x) / width
pixel_height = (2.0 * (fov_y / (width/height))) / height

anti_alias_diff_x = pixel_width * 0.35 
anti_alias_diff_y = pixel_height * 0.35 


camera_dir =  np.array([-1.0, 0.0, 0.0], dtype=np.float32)

init_camera_pos = np.array([20.0, 0.0, 1.0])

camera_pos = np.broadcast_to(init_camera_pos, (5, 3)) + np.array([[0.0, 0.0, 0.0],
                       [0.0,anti_alias_diff_x,0.0],
                       [0.0,-anti_alias_diff_x,0.0],
                       [0.0,0,0.0+anti_alias_diff_y],
                       [0.0,0,0.0-anti_alias_diff_y]], dtype=np.float32)



# 0 < a < 1, 1 is max, 0 is schwarzshield
a = 0.9




# Calxing functions
@njit
def outer_event_horizon_R(a):
     return 1 + np.sqrt(np.maximum(1 - a**2, 0.0))

@njit
def ISCO(a): #M = 1
     Z1 = 1 + ((1 - a**2) ** (1/3)) * ( np.pow((1 + a),1/3) + np.pow((1 - a),1/3) )
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
     
     d_pr = 2*r*(r**2 + a**2 - a*L_z) - (r - 1)*(Q + (L_z - a)**2)
     d_ptheta = np.cos(theta) * (L_z**2 / (np.sin(theta)**3 + 1e-10) - a**2 * np.sin(theta))

     d_r = pr
     d_theta = ptheta
     d_phi = a / (delta(r,a) + 1e-10) * ((r**2 + a**2) - a*L_z) - a + L_z / (np.sin(theta)**2 + 1e-10)

     return d_r, d_theta, d_phi, d_pr, d_ptheta

@njit()
def calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, step): # RK4

    d_r1, d_theta1, d_phi1, d_pr1, d_ptheta1 = calculate_derivatives(r, theta, phi, pr, ptheta, a, L_z, Q)

    r2 = r + d_r1 * step/2
    theta2 = theta + d_theta1 * step/2
    phi2 = phi + d_phi1 * step/2
    pr2 = pr + d_pr1 * step/2
    ptheta2 = ptheta + d_ptheta1 * step/2

    d_r2, d_theta2, d_phi2, d_pr2, d_ptheta2 = calculate_derivatives(r2, theta2, phi2, pr2, ptheta2, a, L_z, Q)

    r3 = r + d_r2 * step/2
    theta3 = theta + d_theta2 * step/2
    phi3 = phi + d_phi2 * step/2
    pr3 = pr + d_pr2 * step/2
    ptheta3 = ptheta + d_ptheta2 * step/2

    d_r3, d_theta3, d_phi3, d_pr3, d_ptheta3 = calculate_derivatives(r3, theta3, phi3, pr3, ptheta3, a, L_z, Q)

    r4 = r + d_r3 * step
    theta4 = theta + d_theta3 * step
    phi4 = phi + d_phi3 * step
    pr4 = pr + d_pr3 * step
    ptheta4 = ptheta + d_ptheta3 * step

    d_r4, d_theta4, d_phi4, d_pr4, d_ptheta4 = calculate_derivatives(r4, theta4, phi4, pr4, ptheta4, a, L_z, Q)

    r_final = r + step/6 * (d_r1 + 2 * d_r2 + 2 * d_r3 + d_r4)
    theta_final = theta + step/6 * (d_theta1 + 2 * d_theta2 + 2 * d_theta3 + d_theta4)
    phi_final = phi + step/6 * (d_phi1 + 2 * d_phi2 + 2 * d_phi3 + d_phi4)
    pr_final = pr + step/6 * (d_pr1 + 2 * d_pr2 + 2 * d_pr3 + d_pr4)
    ptheta_final = ptheta + step/6 * (d_ptheta1 + 2 * d_ptheta2 + 2 * d_ptheta3 + d_ptheta4)
    
    
    return r_final,theta_final,phi_final,pr_final,ptheta_final

@njit
def update(pos,pvelo, inward, L_z, a, Q, dt, r_ISCO):
    r = pos[0]
    theta = pos[1]
    phi = pos[2]
    
    pr = pvelo[0]
    ptheta = pvelo[1]
    
    r_new,theta_new,phi_new,pr_new,ptheta_new = calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, dt)
    pos_new = np.array([r_new,theta_new,phi_new])
    pvelo_new = np.array([pr_new,ptheta_new])
    
    state_new = checkCollided(pos,pos_new,a,r_ISCO)
    
    return pos_new,inward, pvelo_new, state_new

@njit
def checkCollided(prev_pos, pos,a, r_ISCO):
    prev_z = to_cartesian(prev_pos)[2]
    z = to_cartesian(pos)[2]
    r = pos[0]

    horizon = outer_event_horizon_R(a) + 0.02

    if r <= horizon:
         return 1
    if (prev_z * z < 0.0) and (r > r_ISCO) and (r < r_ISCO+accretion_disk_r): #Kinda arbitrary, just testing values for now
         return 2
    if (r > maxx):
         return 3
    return 0


plotter = pv.Plotter()
#camera

ratio = width / height
screen = (-1.0*fov_x, 1.0 * fov_y / ratio, 1.0*fov_x, -1.0 * fov_y / ratio) # left, top, right, bottom
image = np.zeros((height, width, 1))

tau = 0
#number of photons
num = width * height

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
L_z = np.zeros((5,num,))
Q = np.zeros((5,num,))
inward = np.zeros((5,num,2),dtype=bool) #For 'r' and 'theta' 
state = np.zeros((5,num,))
r_ISCO = ISCO(a)

#pos, velo,pvelo, inward, L_z, Q = inits(pos, velo, pvelo, inward, a, L_z, Q, camera_pos)

#fpos = pos + [0, 0, z_center * z_stretch]
#points = pv.PointSet(fpos)
#plotter.add_mesh(points, point_size=5.0, render_points_as_spheres=False, color='blue', name="my_points")

@njit(parallel=True)
def updateParticles(pos, pvelo, L_z, state, inward, Q, r_ISCO):
     for j in range(len(pos)):
         for i in prange(len(pos[0])):
             if state[j,i] == 0:
                 pos_new,inward_new, pvelo_new, state_new = update(pos[j,i], pvelo[j,i], inward[j,i], L_z[j,i], a, Q[j,i], dt, r_ISCO)
                 pos[j,i] = pos_new
                 inward[j,i] = inward_new
                 pvelo[j,i] = pvelo_new
                 state[j,i] = state_new
             #state[j,i] = checkCollided(pos[j,i], a, r_ISCO)
        

##def draw(pos):
##    fpos = pos + [0, 0, z_center * z_stretch]
##    points.points[:] = fpos
##    points.Modified()

def calc_image(state):
    state_alias = np.zeros((num,))
    for i in range(len(state)):
        for j in range(len(state[0])):
            state_alias[j] += state[i][j] / 5
    return state_alias

def print_image(state):
     new_state = calc_image(state)
     image = np.reshape(new_state, (height, width))
     #image = image / 3.0
     plt.imsave('balck.png', image)

@njit
def main_loop(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO):
     for i in range(steps):
        if not np.any(state == 0):
             break
        updateParticles(pos, pvelo, L_z, state, inward, Q, r_ISCO)
        tau += dt
        #if visuals:
        #     draw(pos)
        #     plotter.update()

def simulate_photons(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO):
    tau = 0
    #pos, velo, inward, L_z
    start_time = time.perf_counter()
    pos, velo, pvelo, inward, L_z, Q = inits(pos, velo, pvelo, inward, a, L_z, Q, camera_pos, camera_dir)
    main_loop(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO)

    print("donzo")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f}, Resolution: {width} x {height}, dt: {dt}")
    print_image(state)
    
        
print("helo")

##if visuals:
##
##     x = np.linspace(minx, maxx, 60)
##     y = np.linspace(minx, maxx, 60)
##     X, Y = np.meshgrid(x, y, indexing='ij')
##     Z = potential(X, Y, a)
##
##     grid = pv.StructuredGrid(X, Y, Z * z_stretch)
##     black = pv.Sphere(radius=outer_event_horizon_R(a), center=(0, 0, z_center * z_stretch))
##     plotter.add_mesh(grid, style='wireframe', color='black', line_width=1, name='potential')
##     plotter.add_mesh(black, color='black',name='black hole')
##
##     plotter.camera_position = 'iso'
##     plotter.set_background('white')
##     plotter.ren_win.SetFullScreen(True)
##     plotter.add_key_event('g', simulate_photons)
##     plotter.show(interactive_update=True,full_screen=True)
##
##     plotter.reset_camera()
##else:
simulate_photons(tau, pos, velo, pvelo, inward, L_z, Q, state, r_ISCO)



    
