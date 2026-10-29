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

d_tau = 0.01 #default accuracy
tau = 0.0
steps = 20000
M = 1.0
button_size = 30


#Changables
minx, maxx = -7, 7 #10
init_dir = 180
init_pos = np.array([3.9,3.9 , z_center * z_stretch])

inward = True
end = True
angle = init_dir

# Calxing functions
def potential(x, y, m):
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 * m / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)

def ang_momentum(x, y, alpha, M): #specific ang momentum
    r = np.sqrt(x**2 + y**2)
    return r * np.sin(alpha) / np.sqrt(1 - 2 * M / r)

def calc_rad_velocity(r, L, M, inward):
    b = L
    rad_v = (1 - 2*M/r) * np.sqrt(np.clip((1 - (1 - 2*M / r) * (b**2 / r**2)),0,None))
    return -rad_v if inward else rad_v

def calc_ang_velocity(r, L, M):
    return (1 - 2*M / r) * L / (r**2)

def calculate_new_pos(x, y, M, L, step,inward): # RK4
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    #k1
    rad_v = calc_rad_velocity(r,L,M,inward)
    ang_v = calc_ang_velocity(r,L,M)

    r2 = r + rad_v * step/2
    p2 = phi + ang_v * step/2

    #k2
    rad_v2 = calc_rad_velocity(r2,L,M,inward)
    ang_v2 = calc_ang_velocity(r2,L,M)

    r3 = r + rad_v2 * step/2
    p3 = phi + ang_v2 * step/2

    #k3
    rad_v3 = calc_rad_velocity(r3,L,M,inward)
    ang_v3 = calc_ang_velocity(r3,L,M)

    r4 = r + rad_v3 * step
    p4 = phi + ang_v3 * step

    #k4
    rad_v4 = calc_rad_velocity(r4,L,M,inward)
    ang_v4 = calc_ang_velocity(r4,L,M)

    r_final = r + step/6 * (rad_v + 2 * rad_v2 + 2 * rad_v3 + rad_v4)
    phi_final = phi + step/6 * (ang_v + 2 * ang_v2 + 2 * ang_v3 + ang_v4)

    x_final = r_final * np.cos(phi_final)
    y_final = r_final * np.sin(phi_final)
    return (x_final,y_final)

#Setting up the potential mesh-grid
plotter = pv.Plotter()

x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = potential(X, Y, M)

grid = pv.StructuredGrid(X, Y, Z * z_stretch)
black = pv.Sphere(radius=2.0, center=(0, 0, z_center * z_stretch))


class Photon:
    def __init__(self, init_pos, angle):
        self.init_pos = init_pos
        self.angle = angle
        self.photonCentre = init_pos
        self.pos = pv.Line((0, 0, z_center * z_stretch), self.init_pos)
        self.velocity = pv.Line(init_pos, init_pos)
        self.inward = False
        self.L = 0
        self.end = False

        plotter.add_mesh(self.velocity, color='green', line_width=3)

        line = pv.lines_from_points(np.array([self.init_pos, self.init_pos], dtype=float))
        self.positionLine = plotter.add_mesh(line, color="cyan", line_width=5)

        self.reset_simulation()

    def reset_simulation(self):
        global photon, positions, tau

        self.inward = np.cos(np.deg2rad(self.angle)) < 0
            
        self.photonCentre = self.init_pos
        self.positions = np.array([self.init_pos, self.init_pos], dtype=float)
        self.positionLine.mapper.dataset.copy_from(pv.lines_from_points(self.positions))

        self.L = ang_momentum(self.init_pos[0],self.init_pos[1], self.angle, M)

        self.update(move=False)
    
    def update(self,move=True):
        global tau, d_tau

        if self.end:
            return
        x,y,z = self.photonCentre
        r = np.sqrt(x**2 + y**2)

        if r <= 2 + 0.02:
            if move:
                print("Event Horizon Reached.")
            self.end = True
            return
        if move:
            if (x > maxx or x < minx) or (y > maxx or y < minx):
                print("Particle escaped grid boundary")
                self.end = True
                return
            #Flip logic
            pot = ((1 - 2*M / r) * (self.L**2 / r**2))

            if self.inward and (pot >= 1**2 - 1e-6):
                self.inward = not self.inward
                print(f"Turning point reached at r = {r:.2f}")
        
        xp, yp = calculate_new_pos(x,y,M,self.L,d_tau,self.inward)
        self.photonCentre  = np.array([xp,yp,z_center * z_stretch])

        #draw curvy line
        self.positions = np.vstack([self.positions, np.array([[xp,yp,z_center * z_stretch]])])
        updated = pv.lines_from_points(self.positions)
        self.positionLine.mapper.dataset.copy_from(updated)
    def isEnd(self):
        return self.end



plotter.add_mesh(grid, style='wireframe', color='black', line_width=1, name='potential')
print("helo")
plotter.add_mesh(black, color='black',name='black hole')

plotter.camera_position = 'iso'
plotter.set_background('white')

#x    init_pos[0] = value
#y    init_pos[1] = value
#angle    angle = value % 360




plotter.reset_camera()

tau = 0
numphotons = 12
photons = np.empty((numphotons),dtype=object)
def simulate_photons():
    global tau
    tau = 0
    for i in range(numphotons):
        x = 6-i
        y = 4
        photons[i] = Photon(np.array([x,y, z_center * z_stretch]),  #FIXXX angle )
    
    for i in range(steps):
        for photon in photons:
            if not photon.isEnd():
                photon.update()
        tau += d_tau
        plotter.update()
        
plotter.ren_win.SetFullScreen(True)
plotter.add_key_event('g', simulate_photons)
plotter.show(interactive_update=True,full_screen=True)


    
