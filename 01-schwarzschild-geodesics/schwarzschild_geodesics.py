'''
Interactive simulation of Schwarzschild geodesics

Launch a photon or massive particle near a black hole and watch its trajectory map out alongside its path on a Penrose diagram.

Controls:
Sliders control mass, launch position, direction and speed
Green button starts / stops
Top-right toggle switches between photon and massive particle
R to reset view
Q to quit

Run: python schwarzschild_geodesics.py

'''

import pyvista as pv
import numpy as np
import warnings
import vtk
try:
    import colour
    colour.convert
except (ImportError, AttributeError):
    raise SystemExit("Missing dependency: run  pip install colour-science  (not 'colour')")

# UNITS:
# geometric units: G = c = 1.  r, t, and M share one length unit
# Both photons and massive particles are integrated in Schwarzschild coordinate time, which is also the t used for the Penrose diagram


# Suppress warnings for invalid math in the event horizon
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Parameters
minz, maxz = 0.90, 1.07
z_center = (maxz + minz) / 2
init_mass = 1.0
z_stretch = 8.0

d_t_coord = 0.01 #default accuracy
t_coord = 0.0
orig_wavelength = 450
steps = 150000
M = init_mass
button_size = 30

#Penrose diagram variables
n_points = 40
amplitude = 0.05

t = np.linspace(-200.,200.,400)
tp = np.linspace(-200.,200,750,endpoint=False)[1:]
r = np.concatenate((2.0 - np.logspace(-4, np.log10(1.99), 25),2.0 + np.logspace(-4, 4, 50)))
rp = np.concatenate((2.0 + np.logspace(-4, 1, 25), 2.0 - np.logspace(-4, np.log10(1.99), 25)))
rp.sort()

top    = [1,  1, 0]
bottom = [1, -1, 0]
right  = [2, 0,  0]
left   = [0, 0, 0]
leftest =[-1, 1, 0]

#Zigzag at singularity
sx = np.linspace(-1, 1, n_points)
sy = np.zeros(n_points)
sy[::2] = 1+amplitude/2
sy[1::2] = 1-amplitude/2
sz = np.zeros(n_points)
spoints = np.column_stack((sx, sy, sz))

#Additional parameters
minx, maxx = -13, 13

# Default values: launched at r ~ 11.5M, nearly tangential (95 degrees from radial) at v ~ 0.30,
# just below the circular orbit speed -> bound, precessing orbit
init_dir = 95 #launch direction in degrees measured relative to outward radial direction
init_pos = np.array([8.117,8.117 , z_center * z_stretch])
penrose_scale = 1

massive=True
inward = True
turning = False
end = True
angle = init_dir
v=0.29925

STEPS_PER_FRAME = 25

trail = [init_pos.copy(), init_pos.copy()]
trail_times = [0.0, 0.0]                     # coordinate time of each trail point (for colouring)
line = pv.lines_from_points(np.array(trail))
line["time"] = np.array(trail_times)

# Calculating functions
def lapse(x, y, m):
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 * m / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)

def gamma(v):
    return 1 / np.sqrt(1 - v**2)

def Energy(x, y, v, M, massive): #specific energy
    r = np.sqrt(x**2 + y**2)
    if r <= 2 * M + 0.02: 
        return 1.0 
    if massive:
        return np.sqrt(1 - 2*M/r) * gamma(v)
    return 1.0

def ang_momentum(x, y, alpha, M, v, massive): #specific ang momentum
    r = np.sqrt(x**2 + y**2)
    if r <= 2 * M + 0.02: 
        return 0.0 
    if massive:
        return r * np.sin(alpha) * gamma(v) * v
    return r * np.sin(alpha) / np.sqrt(1 - 2 * M / r)

def calc_rad_velocity(r, L, M, E, massive, inward):
    if not massive:
        b = L / E
        rad_v = (1 - 2*M/r) * np.sqrt(np.clip((1 - (1 - 2*M / r) * (b**2 / r**2)),0,None))
    else:
        rad_v = (1 - 2*M/r) * np.sqrt(np.clip((E**2 - (1 - 2*M / r) * (1 + L**2 / r**2)),0,None)) / E
    return -rad_v if inward else rad_v

def calc_ang_velocity(r, L, M, E):
    return (1 - 2*M / r) * L / (E * r**2)

def calculate_new_pos(x, y, M, L, E, step, massive, inward): # RK4
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    #k1
    rad_v = calc_rad_velocity(r,L,M,E,massive,inward)
    ang_v = calc_ang_velocity(r,L,M,E)

    r2 = r + rad_v * step/2
    p2 = phi + ang_v * step/2

    #p2, p3 and p4 unused as derivatives don't depend upon phi

    #k2
    rad_v2 = calc_rad_velocity(r2,L,M,E,massive,inward)
    ang_v2 = calc_ang_velocity(r2,L,M,E)

    r3 = r + rad_v2 * step/2
    p3 = phi + ang_v2 * step/2

    #k3
    rad_v3 = calc_rad_velocity(r3,L,M,E,massive,inward)
    ang_v3 = calc_ang_velocity(r3,L,M,E)

    r4 = r + rad_v3 * step
    p4 = phi + ang_v3 * step

    #k4
    rad_v4 = calc_rad_velocity(r4,L,M,E,massive,inward)
    ang_v4 = calc_ang_velocity(r4,L,M,E)

    r_final = r + step/6 * (rad_v + 2 * rad_v2 + 2 * rad_v3 + rad_v4)
    phi_final = phi + step/6 * (ang_v + 2 * ang_v2 + 2 * ang_v3 + ang_v4)

    x_final = r_final * np.cos(phi_final)
    y_final = r_final * np.sin(phi_final)
    return (x_final,y_final)

def get_rgb(nm):
    rgb = colour.convert(max(360,min(780,nm)), 'Wavelength', 'sRGB')
    return tuple(rgb.clip(0, 1))

# precomputed RGB table
RGB_WAVELENGTHS = np.arange(360, 785, 5)
RGB_TABLE = [get_rgb(nm) for nm in RGB_WAVELENGTHS]

def wavelength_to_rgb(nm):
    index = int(np.clip(round((nm - 360) / 5), 0, len(RGB_TABLE) - 1))
    return RGB_TABLE[index]

#Penrose Diagram
class PD:

    #Converts schwarzschild spacetime coords (r,t) to compactified penrose coords
    @staticmethod
    def penrose_coords(r_in,t_in,roff=0,toff=0):
        r = np.atleast_1d(np.array(r_in,dtype="float"))
        t = np.atleast_1d(np.array(t_in,dtype="float"))

        if t.size == 1 and r.size>1:
            t = np.full_like(r,t[0])
        if r.size == 1 and t.size>1:
            r = np.full_like(t,r[0])

        VplusU, VminusU = np.zeros_like(r), np.zeros_like(r)

        ext = r > 2   # outside the horizon
        if np.any(ext):
            c_ext = np.sqrt(r[ext]/2 - 1) * np.exp(r[ext]/4)
            VplusU[ext]  =  c_ext * np.exp(t[ext]/4)
            VminusU[ext] = -c_ext * np.exp(-t[ext]/4)

        inte = r < 2  # inside the horizon
        if np.any(inte):
            c_int = np.sqrt(1 - r[inte]/2) * np.exp(r[inte]/4)
            VplusU[inte]  = c_int * np.exp(t[inte]/4)
            VminusU[inte] = c_int * np.exp(-t[inte]/4)

        VplusU /= penrose_scale
        VminusU /= penrose_scale

        Tp = 2./np.pi*(np.arctan(VplusU) + np.arctan(VminusU))
        Rp = 2./np.pi*(np.arctan(VplusU) - np.arctan(VminusU))
        return Rp + roff,Tp + toff

    #Draws the diagram grid of lines of constant t and r and the photon sphere, on a separate static renderer
    @staticmethod
    def draw_diagram(t,r,tp,rp,roff=0,toff=0,mint=-50,maxt=50,minr=0.01,maxr=50):
        t = t[(t >= mint) & (t <= maxt)]
        tp = tp[(tp >= mint) & (tp <= maxt)]
        r = r[(r >= minr) & (r <= maxr)]
        rp = rp[(rp >= minr) & (rp <= maxr)]

        curves = []
        for t_val in t:
            R,T = PD.penrose_coords(rp,t_val,roff=roff,toff=toff)
            curves.append(np.column_stack((R, T, np.zeros_like(R))))
        for r_val in r:
            R,T = PD.penrose_coords(r_val,tp,roff=roff,toff=toff)
            curves.append(np.column_stack((R, T, np.zeros_like(R))))
        PD.add_obj(PD.curves_to_mesh(curves), color="black")

        #photon sphere
        R,T = PD.penrose_coords(3.0,tp,roff=roff,toff=toff)
        PD.add_obj(pv.MultipleLines(np.column_stack((R, T, np.zeros_like(R)))), color="red")

    @staticmethod
    def curves_to_mesh(curves):
        # Combine many polylines into one mesh
        points = np.vstack(curves)
        cells, offset = [], 0
        for c in curves:
            cells.append(np.concatenate(([len(c)], np.arange(offset, offset + len(c)))))
            offset += len(c)
        return pv.PolyData(points, lines=np.concatenate(cells))

    @staticmethod
    def add_obj(obj, color="green"):
        actor = plotter.add_mesh(obj, color=color, line_width=2)
        plotter.renderer.RemoveActor(actor) 
        static_renderer.AddActor(actor)


    @staticmethod
    def draw(t,r,tp,rp):
        PD.draw_diagram(t,r,tp,rp,roff=0,toff=0,minr=0.01,maxr=40,mint=-40,maxt=40)
        static_renderer.ResetCamera()
        
        boundary_points = np.array([bottom, right, top, left, bottom])
        triangle = np.array([top, left, leftest])
        lines1 = pv.MultipleLines(points=boundary_points)
        lines2 = pv.MultipleLines(points=triangle)
        zigzag = pv.MultipleLines(spoints)

        PD.add_obj(lines1,color="green")
        PD.add_obj(lines2,color="green")
        PD.add_obj(zigzag,color="red")

#Setting up the potential mesh-grid
plotter = pv.Plotter(window_size=(1600, 900))

x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = lapse(X, Y, init_mass)

L = ang_momentum(init_pos[0],init_pos[1], np.deg2rad(init_dir), M, v, massive)
E = Energy(init_pos[0],init_pos[1], v, M, massive)

grid = pv.StructuredGrid(X, Y, Z * z_stretch)
black = pv.Sphere(radius=2.0, center=(0, 0, z_center * z_stretch))
photon_sphere = pv.Disc(center=(0, 0, z_center * z_stretch), inner=2.9, outer=3.0,c_res=70)

photon = pv.Sphere(radius=0.1, center=init_pos)
pos = pv.Line((0, 0, z_center * z_stretch), init_pos)
velocity = pv.Line(init_pos, init_pos)

#Second renderer for penrose diagram (so that it isn't altered by camera view
static_renderer = vtk.vtkRenderer()
vx0, vy0, vx1, vy1 = 0.60, 0.18, 0.98, 0.60
static_renderer.SetViewport(vx0, vy0, vx1, vy1)
static_renderer.SetLayer(1)
static_renderer.InteractiveOff()
plotter.ren_win.SetNumberOfLayers(2)
plotter.ren_win.AddRenderer(static_renderer)

#Setting up actors for all the bodies
plotter.add_mesh(grid, style='wireframe', color='black', line_width=1, name='potential')
plotter.add_mesh(black, color='black',name='black hole')
plotter.add_mesh(photon_sphere, color="red",name='photon_sphere')
photon_actor = plotter.add_mesh(photon, color=get_rgb(orig_wavelength),name='photon')
plotter.add_mesh(pos, color='black', line_width=3)
plotter.add_mesh(velocity, color='green', line_width=3)
positionLine = plotter.add_mesh(line, scalars="time", cmap="cool", line_width=5, show_scalar_bar=False)

PD.draw(t,r,tp,rp)

position = pv.Sphere(radius=0.05,center=(0,0,0))
diagram = plotter.add_mesh(position, color="yellow", line_width=2)
plotter.renderer.RemoveActor(diagram) 
static_renderer.AddActor(diagram)

static_renderer.ResetCamera()
plotter.camera_position = 'iso'
plotter.set_background('white')

def update_diagram(r,t):
    R,T = PD.penrose_coords(r,t,0,0)
    target = np.array([R[0],T[0],0.0],dtype=float)
    position.translate(target - position.center, inplace=True)

def step_physics():
    # Advance the particle by one RK4 step
    global inward, end, t_coord, turning, particle, last_velo
    x, y = particle
    r = np.hypot(x, y)
    E_eff = E if massive else 1.0

    if r <= 2 * M + 0.02: #Stops integration slightly outside to avoid coordinate singularity
        print("[INFO] Event Horizon Reached.")
        end = True
        return
    if not (minx <= x <= maxx and minx <= y <= maxx):
        print("[INFO] Particle escaped grid boundary")
        end = True
        return

    #Flip logic
    pot = (1 - 2*M / r) * ((1.0 if massive else 0.0) + L**2 / r**2)
    if pot >= E_eff**2 - 1e-6:
        if not turning:
            inward = not inward
            turning = True
            print(f"[INFO] Turning point at r = {r:.2f}")
    else:
        turning = False

    t_coord += d_t_coord
    xp, yp = calculate_new_pos(x, y, M, L, E, d_t_coord, massive, inward)
    last_velo = np.array([xp - x, yp - y, 0.0])
    particle = np.array([xp, yp])


def refresh_visuals():
    x, y = particle
    r = np.hypot(x, y)
    centre = np.array([x, y, z_center * z_stretch])

    photon.translate(centre - np.array(photon.center), inplace=True)
    pos.points[1] = centre
    velocity.points[0] = centre
    norm = np.linalg.norm(last_velo)
    velocity.points[1] = centre + last_velo * 3 / norm if norm > 1e-8 else centre

    trail.append(centre)
    trail_times.append(t_coord)
    trail_mesh = pv.lines_from_points(np.array(trail))
    trail_mesh["time"] = np.array(trail_times)
    positionLine.mapper.dataset.copy_from(trail_mesh)
    positionLine.mapper.scalar_range = (0.0, max(t_coord, 1e-6))

    update_diagram(r / M, t_coord / M)

    if massive:
        photon_actor.prop.color = "Green"
    else:
        #Calculate colour of photon (from an observer at infinity)
        g_tt = max(1e-6, 1 - 2*M/r)                    #Ensure no infinite wavelengths
        photon_actor.prop.color = wavelength_to_rgb(orig_wavelength / np.sqrt(g_tt))

#Sliders and callbacks
def reset_simulation():
    global L, E, particle, trail, last_velo, t_coord, inward, turning, trail_times

    inward = np.cos(np.deg2rad(angle)) < 0
    turning = False
    t_coord = 0.0

    particle = init_pos[:2].copy()
    trail = [init_pos.copy(), init_pos.copy()]

    L = ang_momentum(init_pos[0],init_pos[1],np.deg2rad(angle), M, v, massive)
    E = Energy(init_pos[0],init_pos[1], v, M, massive)

    r = max(np.hypot(*particle), 1e-9)
    radial = np.array([particle[0], particle[1], 0.0]) / r
    tangential = np.array([-particle[1], particle[0], 0.0]) / r
    alpha = np.deg2rad(angle)
    last_velo = np.cos(alpha) * radial + np.sin(alpha) * tangential

    trail = [init_pos.copy(), init_pos.copy()]
    trail_times = [0.0, 0.0]

    refresh_visuals()

def update_mass(value):
    global M
    if not end:
        try: mass_slider.GetRepresentation().SetValue(M); return #Prevent slider changes while the simulation is running
        except NameError: return
    M = value
    new_z = lapse(X, Y, value)
    grid.points[:, 2] = new_z.ravel() * z_stretch

    new_sphere = pv.Sphere(radius=2*value, center=(0, 0, z_center * z_stretch))
    plotter.add_mesh(new_sphere, color="black", name="black hole")
    new_ring = pv.Disc(center=(0, 0, z_center * z_stretch), inner=3.0*value - 0.1, outer=3.0*value,c_res=70)
    plotter.add_mesh(new_ring, color="red",name='photon_sphere')
    reset_simulation()

def update_x(value):
    if not end:
        try: x_slider.GetRepresentation().SetValue(init_pos[0]); return #Prevent slider changes while the simulation is running
        except NameError: return
    init_pos[0] = value
    reset_simulation()

def update_y(value):
    if not end:
        try: y_slider.GetRepresentation().SetValue(init_pos[1]); return
        except NameError: return
    init_pos[1] = value
    reset_simulation()

def update_angle(value):
    global angle
    if not end:
        try: angle_slider.GetRepresentation().SetValue(angle); return
        except NameError: return
    angle = value % 360
    reset_simulation()    

def update_massive(value):
    global massive
    if not end:
        try: massive_button.GetRepresentation().SetState((not value)); return
        except NameError: return
    massive = value
    reset_simulation()

def update_velo(value):
    global v
    if not massive or not end:
        try: velo_slider.GetRepresentation().SetValue(v)
        except NameError: pass
        return
    v = value
    reset_simulation()

def reset_view():
    if not end:
        return
    update_mass(init_mass)
    try: mass_slider.GetRepresentation().SetValue(init_mass)
    except NameError: pass
    
def start_simulation(value):
    global end
    if value:
        end = False
        reset_simulation()
        for frame in range(steps // STEPS_PER_FRAME):
            for _ in range(STEPS_PER_FRAME):
                step_physics()
                if end: break
            refresh_visuals()
            try:
                plotter.update()
            except (RuntimeError, AttributeError):
                return  # window closed
            if end: break
        end = True
        reset_simulation()
        try: start_button.GetRepresentation().SetState(0)
        except NameError: pass
    else:
        end = True
        reset_simulation()
        
#Pyvista UI - Handling sliders and button

mass_slider  = plotter.add_slider_widget(callback=update_mass, rng=[0.1, 4.0], value=init_mass, title="Mass", pointa=(0.02, 0.1), pointb=(0.24, 0.1), style='modern', color='black')
x_slider     = plotter.add_slider_widget(callback=update_x, rng=[minx, maxx], value=init_pos[0], title="Initial X", pointa=(0.26, 0.1), pointb=(0.48, 0.1), style='modern', color='black')
y_slider     = plotter.add_slider_widget(callback=update_y, rng=[minx, maxx], value=init_pos[1], title="Initial Y", pointa=(0.50, 0.1), pointb=(0.72, 0.1), style='modern', color='black')
angle_slider = plotter.add_slider_widget(callback=update_angle, rng=[0, 360], value=angle, title="Initial Direction", pointa=(0.74, 0.1), pointb=(0.96, 0.1), style='modern', color='black')
velo_slider  = plotter.add_slider_widget(callback=update_velo, rng=[0.05, 0.999], value=v, title="Velocity", pointa=(0.74, 0.8), pointb=(0.96, 0.8), style='modern', color='black')

start_button = plotter.add_checkbox_button_widget(start_simulation, value=False, position=(20, 150), size=button_size, color_on='green', color_off='grey')
plotter.add_text("Start/Reset Simulation", position=(20 + button_size + 10, 152), font_size=10, color="black", name="button_label")


massive_button = plotter.add_checkbox_button_widget(update_massive, value=True, position=(plotter.window_size[0] - 425, plotter.window_size[1] - 30), size=button_size, color_on='Purple', color_off='Green')
plotter.add_text("Toggle photon/massive     ", position='upper_right', font_size=10, color="black", name="button2_label")

W, H = plotter.window_size
plotter.add_text("Penrose Diagram", position=(int((vx0 + vx1) / 2 * W) - 75, int(vy1 * H) - 75),font_size=10, color="black", name="PD_label")

plotter.add_text("R: reset view    Q: quit", position='upper_left', color='black', font_size=10)
plotter.add_key_event('r', reset_view)

reset_simulation()

plotter.show()


    
