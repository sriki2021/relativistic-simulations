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

# Calxing functions
def potential(x, y):
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)

def ang_momentum(pos, v): #specific ang momentum
    #r = np.sqrt(x**2 + y**2)
    #return r * np.sin(alpha) / np.sqrt(1 - 2 * M / r)
    return (pos[:,0]*v[:,1] - pos[:,1]*v[:,0])/np.linalg.norm(v, axis=1)

def calc_rad_velocity(r, L, inward):
    rad_v = (1 - 2/r) * np.sqrt(np.clip((1 - (1 - 2 / r) * (L**2 / r**2)),0,None))
    return np.where(inward, -rad_v, rad_v)

def calc_ang_velocity(r, L):
    return (1 - 2 / r) * L / (r**2)

def calculate_new_pos(r, phi, L, step,inward): # RK4

    #k1
    rad_v = calc_rad_velocity(r,L,inward)
    ang_v = calc_ang_velocity(r,L)

    r2 = r + rad_v * step/2
    p2 = phi + ang_v * step/2

    #k2
    rad_v2 = calc_rad_velocity(r2,L,inward)
    ang_v2 = calc_ang_velocity(r2,L)

    r3 = r + rad_v2 * step/2
    p3 = phi + ang_v2 * step/2

    #k3
    rad_v3 = calc_rad_velocity(r3,L,inward)
    ang_v3 = calc_ang_velocity(r3,L)

    r4 = r + rad_v3 * step
    p4 = phi + ang_v3 * step

    #k4
    rad_v4 = calc_rad_velocity(r4,L,inward)
    ang_v4 = calc_ang_velocity(r4,L)

    r_final = r + step/6 * (rad_v + 2 * rad_v2 + 2 * rad_v3 + rad_v4)
    phi_final = phi + step/6 * (ang_v + 2 * ang_v2 + 2 * ang_v3 + ang_v4)

    x_final = r_final * np.cos(phi_final)
    y_final = r_final * np.sin(phi_final)
    return np.column_stack((x_final,y_final))

def update(pos,inward, L, dt):
    r = np.linalg.norm(pos, axis=1) #np.sqrt(pos[:].dot(pos[:]))
    phi = np.arctan2(pos[:,1],pos[:,0])

    pot = ((1 - 2 / r) * (L**2 / r**2))
    mask = (inward) & (pot >= 1**2 - 1e-6)
    inward[mask] = ~inward[mask]

    
    
    pos = calculate_new_pos(r,phi,L,dt,inward)
    return pos,inward

def checkCollided(pos):
    x = pos[:,0]
    y = pos[:,1]
    r = np.sqrt(x**2 + y**2)
    end = np.zeros(len(pos))
    end[r <= 2 + 0.02] = 1
    end[((x > maxx) | (x < minx)) | ((y > maxx) | (y < minx))] = 2
    return end



#Setting up the potential mesh-grid
plotter = pv.Plotter()

x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = potential(X, Y)

grid = pv.StructuredGrid(X, Y, Z * z_stretch)
black = pv.Sphere(radius=2.0, center=(0, 0, z_center * z_stretch))

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
    velo[:, 0] = 0
    velo[:, 1] = -1

    dots = np.sum(pos * velo, axis=1)
    inward = dots < 0
    
    L = ang_momentum(pos, velo)
    return pos, velo, inward, L

plotter.reset_camera()

tau = 0
num = 20

pos = np.zeros((num,2))
velo = np.zeros((num,2))
L = np.zeros((num,))
inward = np.zeros((num,),dtype=bool)
state = np.zeros((num,))

pos, velo, inward, L = inits(pos, velo, inward)

zs = np.full((pos.shape[0], 1),z_center * z_stretch)
fpos = np.hstack((pos, zs))

points = pv.PointSet(fpos)

plotter.add_mesh(points, point_size=5.0, render_points_as_spheres=False, color='blue', name="my_points")

def updateParticles(pos, L, state, inward):
    mask = state == 0
    pos[mask],inward[mask] = update(pos[mask], inward[mask], L[mask], dt)
    state[:] = checkCollided(pos)


def draw(pos):
    zs = np.full((pos.shape[0], 1),z_center * z_stretch)
    fpos = np.hstack((pos, zs))
    points.points[:] = fpos
    points.Modified()

def simulate_photons():
    global tau, pos, velo, inward, L, state
    tau = 0
    #pos, velo, inward, L = inits(pos, velo, inward)
    
    for i in range(steps):
        updateParticles(pos, L, state, inward)
        tau += dt
        draw(pos)
        plotter.update()
        
plotter.ren_win.SetFullScreen(True)
plotter.add_key_event('g', simulate_photons)
plotter.show(interactive_update=True,full_screen=True)


    
