import pyvista as pv
import numpy as np
import warnings
import colour #pip install colour-science
import vtk

# Suppress warnings for invalid math in the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

#Parameters
minz, maxz = 0.90, 1.07
z_center = (maxz + minz) / 2
z_stretch = 8.0

dt = 0.01 #default accuracy
tau = 0.0
steps = 20000
button_size = 30


#Changables
minx, maxx = -7, 7 #10 then 7
init_dir = 180
init_pos = np.array([3.9,3.9 , z_center * z_stretch])

inward = True
end = True
angle = init_dir


# 0 < a < 1, 1 is max, 0 is schwarshield
a = 0.5




# Calxing functions
def outer_event_horizon_R(a):
     return 1 + np.sqrt(np.clip(1 - a**2, 0, None))

def potential(x, y):
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)

def to_spherical(pos):
    w2 = np.sum(pos**2, axis=1) #x**2 + y**2 + z**2
    r = np.sqrt( np.clip( (w2 - a**2 + np.sqrt(np.clip((w2 - a**2)**2 + 4 * a**2 * pos[:,2]**2, 0, None))) / 2, 0, None))
    theta = np.arccos(pos[:,2] / r)
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

def calc_rad_velocity(r, L_z, inward, a, Q):
    rad_v = np.sqrt(np.clip(((r**2 + a**2) - a*L_z)**2 - delta(r,a)*(Q + (L_z - a)**2), 0, None))
    return np.where(inward[:,0], -rad_v, rad_v)

def calc_theta_velocity(theta, L_z, inward, a, Q):
    theta_v = np.sqrt(np.clip((Q + a**2 * np.cos(theta)**2 - L_z**2 * (np.cos(theta)/np.sin(theta))**2),0,None))
    return np.where(inward[:,1], -theta_v, theta_v)

def calc_phi_velocity(theta, L_z, a, r):
    phi_v = a / delta(r,a) * ((r**2 + a**2) - a*L_z) - a + L_z / np.sin(theta)**2
    return phi_v

def calculate_new_pos(r, theta, phi, L_z, a, Q, inward, step): # RK4

    #k1
    rad_v = calc_rad_velocity(r, L_z, inward, a, Q)
    theta_v = calc_theta_velocity(theta, L_z, inward, a, Q)
    phi_v = calc_phi_velocity(theta, L_z, a, r)

    r2 = r + rad_v * step/2
    theta2 = theta + theta_v * step/2
    phi2 = phi + phi_v * step/2

    #k2
    rad_v2 = calc_rad_velocity(r2, L_z, inward, a, Q)
    theta_v2 = calc_theta_velocity(theta2, L_z, inward, a, Q)
    phi_v2 = calc_phi_velocity(theta2, L_z, a, r2)

    r3 = r + rad_v2 * step/2
    theta3 = theta + theta_v2 * step/2
    phi3 = phi + phi_v2 * step/2

    #k3
    rad_v3 = calc_rad_velocity(r3, L_z, inward, a, Q)
    theta_v3 = calc_theta_velocity(theta3, L_z, inward, a, Q)
    phi_v3 = calc_phi_velocity(theta3, L_z, a, r3)

    r4 = r + rad_v3 * step
    theta4 = theta + theta_v3 * step
    phi4 = phi + phi_v3 * step

    #k4
    rad_v4 = calc_rad_velocity(r4, L_z, inward, a, Q)
    theta_v4 = calc_theta_velocity(theta4, L_z, inward, a, Q)
    phi_v4 = calc_phi_velocity(theta4, L_z, a, r4)

    r_final = r + step/6 * (rad_v + 2 * rad_v2 + 2 * rad_v3 + rad_v4)
    theta_final = theta + step/6 * (theta_v + 2 * theta_v2 + 2 * theta_v3 + theta_v4)
    phi_final = phi + step/6 * (phi_v + 2 * phi_v2 + 2 * phi_v3 + phi_v4)
    
    return r_final,theta_final,phi_final

def update(pos,inward, L_z, a, Q, dt):
    r, theta, phi = to_spherical(pos)
    
    Pot_R = ((r**2 + a**2) - a*L_z)**2 - delta(r,a)*(Q + (L_z - a)**2)
    
    mask1 = (inward[:,0]) & (Pot_R <= 1e-6)
    inward[mask1,0] = ~inward[mask1,0]

    Pot_Theta = (Q + a**2 * np.cos(theta)**2 - L_z**2 * (np.cos(theta)/np.sin(theta))**2)
    mask2 = (inward[:,1]) & (Pot_Theta <= 1e-6)
    inward[mask2,1] = ~inward[mask2,1]
    
    r_new,theta_new,phi_new = calculate_new_pos(r, theta, phi, L_z, a, Q, inward, dt)
    pos_new = to_cartesian(r_new,theta_new,phi_new)
    
    return pos_new,inward

def checkCollided(pos,a):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    end = np.zeros(len(pos))
    end[r <= outer_event_horizon_R(a) + 0.02] = 1
    end[((x > maxx) | (x < minx)) | ((y > maxx) | (y < minx))] = 2
    return end



#Setting up the potential mesh-grid
plotter = pv.Plotter()

x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = potential(X, Y)

grid = pv.StructuredGrid(X, Y, Z * z_stretch)
black = pv.Sphere(radius=outer_event_horizon_R(a), center=(0, 0, z_center * z_stretch))

##
##class Photon:
##    def __init__(self, init_pos, v):
##        self.init_pos = init_pos
##        self.v = v
##        self.position = init_pos
##        self.pos = pv.Line((0, 0, z_center * z_stretch), self.init_pos)
##        self.velocity = pv.Line(init_pos, init_pos)
##        self.inward = False
##        self.L = 0
##        self.end = False
##
##        plotter.add_mesh(self.velocity, color='green', line_width=3)
##
##        line = pv.lines_from_points(np.array([self.init_pos, self.init_pos], dtype=float))
##        self.positionLine = plotter.add_mesh(line, color="cyan", line_width=3)
##
##        self.reset_simulation()
##
##    def reset_simulation(self):
##        global photon, positions, tau
##
##        self.inward = self.v.dot(self.position[:2]) < 0
##
##        self.position = self.init_pos
##        self.positions = np.array([self.init_pos, self.init_pos], dtype=float)
##        self.positionLine.mapper.dataset.copy_from(pv.lines_from_points(self.positions))
##
##        self.L = ang_momentum(self.init_pos[0],self.init_pos[1], self.v)
##
##        self.update(self.pos,self.L,dt,self.inward,move=False)
##
##
##
##            
##        
##    def isEnd(self):
##        return self.end
##
##

plotter.add_mesh(grid, style='wireframe', color='black', line_width=1, name='potential')
print("helo")
plotter.add_mesh(black, color='black',name='black hole')

plotter.camera_position = 'iso'
plotter.set_background('white')

def inits(pos, velo, inward):
    i = np.arange(num)
    pos[:, 0] = 6 - 0.5*i
    pos[:, 1] = 3 + 0.15*i
    pos[:, 2] = 3 + 0.1*i
    velo[:, 0] = 0
    velo[:, 1] = -1

    dots = np.sum(pos * velo, axis=1)
    inward[:,0] = dots < 0
    inward[:,1] = velo[:, 2] > 0 
    
    L_z = axial_ang_momentum(pos, velo)

    r,theta,phi = to_spherical(pos)
    Q = Carter_Constant(L_z, a, velo, r, theta, phi)
    return pos, velo, inward, L_z, Q

plotter.reset_camera()

tau = 0
num = 20

pos = np.zeros((num,3))
velo = np.zeros((num,3))
L_z = np.zeros((num,))
Q = np.zeros((num,))
inward = np.zeros((num,2),dtype=bool) #For 'r' and 'theta' 
state = np.zeros((num,))

pos, velo, inward, L_z, Q = inits(pos, velo, inward)

fpos = pos + [0, 0, z_center * z_stretch]
points = pv.PointSet(fpos)
plotter.add_mesh(points, point_size=5.0, render_points_as_spheres=False, color='blue', name="my_points")

def updateParticles(pos, L_z, state, inward, Q):
    mask = state == 0
    pos[mask],inward[mask] = update(pos[mask], inward[mask], L_z[mask], a, Q[mask], dt)
    state[:] = checkCollided(pos, a)


def draw(pos):
    fpos = pos + [0, 0, z_center * z_stretch]
    points.points[:] = fpos
    points.Modified()

def simulate_photons():
    global tau, pos, velo, inward, L_z, state
    tau = 0
    #pos, velo, inward, L_z
    pos, velo, inward, L_z, Q = inits(pos, velo, inward)
    
    for i in range(steps):
        updateParticles(pos, L_z, state, inward, Q)
        tau += dt
        draw(pos)
        plotter.update()
        
plotter.ren_win.SetFullScreen(True)
plotter.add_key_event('g', simulate_photons)
plotter.show(interactive_update=True,full_screen=True)


    
