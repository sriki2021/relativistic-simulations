import pyvista as pv
import numpy as np
import warnings
import colour #pip install colour-science
import vtk
import matplotlib.pyplot as plt
import time

# Suppress warnings for invalid math in the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

#Parameters
minz, maxz = 0.90, 1.07
z_center = (maxz + minz) / 2
z_stretch = 10.0
tau = 0.0

dt = 0.005 #default accuracy
steps = 20000

minx, maxx = -10, 10 #10 then 7

end = True

#CAMERA VARS

width = 300
height = 300

visuals = False

camera_dir =  np.array([-1.0, 0.0, 0.0], dtype=np.float32)
camera_pos = np.array([10.0, 0.0, 2.0], dtype=np.float32)

# 0 < a < 1, 1 is max, 0 is schwarzshield
a = 0.0




# Calxing functions
def outer_event_horizon_R(a):
     return 1 + np.sqrt(np.clip(1 - a**2, 0, None))

def ISCO(a): #M = 1
     Z1 = 1 + np.pow((1 - a**2),1/3) * ( np.pow((1 + a),1/3) + np.pow((1 - a),1/3) )
     Z2 = np.sqrt(3*a**2 + Z1**2)

     r_ISCO = 3 + Z2 - np.sqrt( (3 - Z1) * (3 + Z1 + 2*Z2) )
     return r_ISCO

def potential(x, y, a):
    r = np.sqrt(x**2 + y**2)
    v = -r/(r**2 + a**2)
    return v + 1 #(Constant to fit the graph)

def to_spherical(pos):
    w2 = np.sum(pos**2, axis=1) #x**2 + y**2 + z**2
    r = np.sqrt( np.clip( (w2 - a**2 + np.sqrt(np.clip((w2 - a**2)**2 + 4 * a**2 * pos[:,2]**2, 0, None))) / 2, 0, None))
    theta = np.arccos(np.clip(pos[:,2] / r, -1.0, 1.0)) #clip to avoid errors
    phi = np.arctan2(pos[:,1],pos[:,0])
    return r,theta,phi

def to_cartesian(r,theta,phi):
    x = np.sqrt(np.clip((r**2+a**2),0,None)) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(np.clip((r**2+a**2),0,None)) * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x,y,z))

def latitudinal_momentum(v, r, theta, phi):  
    p_theta = (v[:,0] * np.cos(theta) * np.cos(phi)) + (v[:,1] * np.cos(theta) * np.sin(phi)) + (-v[:,2] * np.sin(theta))
    return p_theta * r

def Carter_Constant(L_z, a, v, r, theta, phi):
    L_theta = latitudinal_momentum(v, r, theta, phi)
    Q = L_theta**2 + np.cos(theta)**2 * (L_z**2 / np.sin(theta)**2 - a**2)
    return Q

def delta(r,a):
    return r**2 + a**2 - 2*r

def axial_ang_momentum(pos, v): #specific ang momentum
    return (pos[:,0]*v[:,1] - pos[:,1]*v[:,0])/np.linalg.norm(v, axis=1)

def calculate_derivatives(r, theta, phi, pr, ptheta, a, L_z, Q):
     #Suspicious convention where pr and ptheta represent derivatives of p and theta (w.r.t mino time)
     #but only the terms d_r and d_theta are used to calculate r and theta (so that they can update independantly)
     #d_pr and d_ptheta represent second derivatives of r and theta

     #phi doesn't need 2nd derivatives as phi doesnt depend on previous d_phi
     
     d_pr = 2*r*(r**2 + a**2 - a*L_z) - (r - 1)*(Q + (L_z - a)**2)
     d_ptheta = np.cos(theta) * (L_z**2 / np.sin(theta)**3 - a**2 * np.sin(theta))

     d_r = pr
     d_theta = ptheta
     d_phi = a / delta(r,a) * ((r**2 + a**2) - a*L_z) - a + L_z / np.sin(theta)**2

     return d_r, d_theta, d_phi, d_pr, d_ptheta

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

def update(pos,pvelo, inward, L_z, a, Q, dt):
    r, theta, phi = to_spherical(pos)
    pr = pvelo[:,0]
    ptheta = pvelo[:,1]
    
    r_new,theta_new,phi_new,pr_new,ptheta_new = calculate_new_pos(r, theta, phi, pr, ptheta, L_z, a, Q, inward, dt)
    pos_new = to_cartesian(r_new,theta_new,phi_new)
    pvelo_new = np.column_stack((pr_new,ptheta_new))
    
    
    return pos_new,inward, pvelo_new

def checkCollided(pos,a, r_ISCO):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    end = np.zeros(len(pos))
    end[r <= outer_event_horizon_R(a) + 0.02] = 1
    end[(np.abs(z) < 0.01) & (r > r_ISCO) & (r < r_ISCO+2)] = 2 #Kinda arbitrary, just testing values for now
    end[((x > maxx) | (x < minx)) | ((y > maxx) | (y < minx)) | ((z > maxx) | (z < minx))] = 3  
    return end

#camera

ratio = width / height
screen = (-1.0, 1.0 / ratio, 1.0, -1.0 / ratio) # left, top, right, bottom
image = np.zeros((height, width, 1))

def inits(pos, velo, inward):
    i = np.linspace(screen[1], screen[3], height)
    pos[:] = camera_pos
    
    velo[:, 0] = -1
    velo[:, 1] = np.tile(np.linspace(screen[0], screen[2], width),height) #tiled
    velo[:, 2] = np.repeat(np.linspace(screen[1], screen[3], height),width) #repeated
    
    dots = np.sum(pos * velo, axis=1)
    inward[:,0] = dots < 0
    inward[:,1] = velo[:, 2] > 0 
    
    L_z = axial_ang_momentum(pos, velo)

    r,theta,phi = to_spherical(pos)
    Q = Carter_Constant(L_z, a, velo, r, theta, phi)

    rad_v = np.sqrt(np.clip(((r**2 + a**2) - a*L_z)**2 - delta(r,a)*(Q + (L_z - a)**2), 0, None))
    pvelo[:,0] = np.where(inward[:,0], -rad_v, rad_v)

    cot_sq = (np.cos(theta) / (np.sin(theta) + 1e-12))**2 #avoid errors
    theta_v = np.sqrt(np.clip((Q + a**2 * np.cos(theta)**2 - L_z**2 * cot_sq), 0, None))
    pvelo[:,1] = np.where(inward[:,1], -theta_v, theta_v)
    return pos, velo, pvelo, inward, L_z, Q

plotter.reset_camera()

tau = 0
#number of photons
num = width * height

pos = np.zeros((num,3))
velo = np.zeros((num,3))
pvelo = np.zeros((num,2))
L_z = np.zeros((num,))
Q = np.zeros((num,))
inward = np.zeros((num,2),dtype=bool) #For 'r' and 'theta' 
state = np.zeros((num,))
r_ISCO = ISCO(a)

pos, velo,pvelo, inward, L_z, Q = inits(pos, velo, inward)

fpos = pos + [0, 0, z_center * z_stretch]
points = pv.PointSet(fpos)
plotter.add_mesh(points, point_size=5.0, render_points_as_spheres=False, color='blue', name="my_points")

def updateParticles(pos, pvelo, L_z, state, inward, Q):
    mask = state == 0
    pos[mask],inward[mask], pvelo[mask] = update(pos[mask], pvelo[mask], inward[mask], L_z[mask], a, Q[mask], dt)
    state[:] = checkCollided(pos, a, r_ISCO)


def draw(pos):
    fpos = pos + [0, 0, z_center * z_stretch]
    points.points[:] = fpos
    points.Modified()

def print_image(state):
     image = np.reshape(state, (height, width))
     image = image / 3.0
     plt.imsave('balck.png', image)

def simulate_photons():
    global tau, pos, velo, pvelo, inward, L_z, state
    tau = 0
    #pos, velo, inward, L_z
    start_time = time.perf_counter()
    pos, velo, pvelo, inward, L_z, Q = inits(pos, velo, inward)
    
    for i in range(steps):
        if not np.any(state == 0):
             break
        updateParticles(pos, pvelo, L_z, state, inward, Q)
        tau += dt
        if visuals:
             draw(pos)
             plotter.update()
    print("donzo")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Time: {elapsed:.4f}, Resolution: {width} x {height}, dt: {dt}")
    print_image(state)
    
        
print("helo")

if visuals:
     plotter = pv.Plotter()

     x = np.linspace(minx, maxx, 60)
     y = np.linspace(minx, maxx, 60)
     X, Y = np.meshgrid(x, y, indexing='ij')
     Z = potential(X, Y, a)

     grid = pv.StructuredGrid(X, Y, Z * z_stretch)
     black = pv.Sphere(radius=outer_event_horizon_R(a), center=(0, 0, z_center * z_stretch))
     plotter.add_mesh(grid, style='wireframe', color='black', line_width=1, name='potential')
     plotter.add_mesh(black, color='black',name='black hole')

     plotter.camera_position = 'iso'
     plotter.set_background('white')
     plotter.ren_win.SetFullScreen(True)
     plotter.add_key_event('g', simulate_photons)
     plotter.show(interactive_update=True,full_screen=True)
else:
     simulate_photons()



    
