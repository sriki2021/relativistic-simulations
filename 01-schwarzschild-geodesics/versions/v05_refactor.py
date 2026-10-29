import pyvista as pv
import numpy as np
import warnings
import colour #pip install colour-science
import vtk

# Suppress warnings for invalid math in the event horizon
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Parameters
minz, maxz = 0.90, 1.07
z_center = (maxz + minz) / 2
init_mass = 1.0
z_stretch = 8.0

d_tau = 0.01 #default accuracy
tau = 0.0
orig_wavelength = 450
steps = 20000
M = init_mass
button_size = 30

#Penrose diagram variables
res = 15
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

#Changables
minx, maxx = -7, 7 #10
init_dir = 180 #Angle measured relative to the radial v 
init_pos = np.array([3.9,3.9 , z_center * z_stretch])
penrose_scale = 1

massive=True
inward = True
end = True
angle = init_dir
v=0.9

positions = np.array([init_pos, init_pos], dtype=float)
line = pv.lines_from_points(positions)

# Calxing functions
def potential(x, y, m):
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 * m / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)

def gamma(v):
    return 1 / np.sqrt(1 - v**2)

def Energy(x, y, v, M, massive): #specific energy
    if massive:
        r = np.sqrt(x**2 + y**2)
        return np.sqrt(1 - 2*M/r) * gamma(v)
    return 1.0

def ang_momentum(x, y, alpha, M, v, massive): #specific ang momentum
    r = np.sqrt(x**2 + y**2)
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

#Penrose Diagram
class PD:

    #Gives the penrose coordinates (plus an offset)
    @staticmethod
    def penrose_coords(r_in,t_in,roff=0,toff=0):
        r = np.atleast_1d(np.array(r_in,dtype="float"))
        t = np.atleast_1d(np.array(t_in,dtype="float"))

        if t.size == 1 and r.size>1:
            t = np.full_like(r,t[0])
        if r.size == 1 and t.size>1:
            r = np.full_like(t,r[0])

        U,V = np.zeros_like(r),np.zeros_like(r)

        ext = r > 2
        if np.any(ext):
            coff_ext = np.sqrt(r[ext]/(2) - 1) * np.exp(r[ext]/(4))
            U[ext] = coff_ext * np.cosh(t[ext] / (4))
            V[ext] = coff_ext * np.sinh(t[ext] / (4))

        inte = r < 2
        if np.any(inte):
            coff_inte = np.sqrt(1 - r[inte]/(2)) * np.exp(r[inte]/(4))
            U[inte] = coff_inte * np.sinh(t[inte] / (4))
            V[inte] = coff_inte * np.cosh(t[inte] / (4))

        VplusU = (V + U) / penrose_scale
        VminusU = (V - U) / penrose_scale
        
        Tp = 2./np.pi*(np.arctan(VplusU) + np.arctan(VminusU))
        Rp = 2./np.pi*(np.arctan(VplusU) - np.arctan(VminusU))
        return Rp + roff,Tp + toff

    #Draws a diagram given t and r (Range of vals can be chosen)
    @staticmethod
    def draw_diagram(t,r,tp,rp,roff=0,toff=0,mint=-50,maxt=50,minr=0.01,maxr=50):
        t = t[(t >= mint) & (t <= maxt)]
        tp = tp[(tp >= mint) & (tp <= maxt)]
        r = r[(r >= minr) & (r <= maxr)]
        rp = rp[(rp >= minr) & (rp <= maxr)]
        
        for t_val in t:
            R,T = PD.penrose_coords(rp,t_val,roff=roff,toff=toff)
            points = np.column_stack((R, T, np.zeros_like(R)))
            line = pv.MultipleLines(points)
            acteur = plotter.add_mesh(line,smooth_shading=True, color="black", line_width=2)
            plotter.renderer.RemoveActor(acteur) 
            static_renderer.AddActor(acteur)
            
        for r_val in r:
            R,T = PD.penrose_coords(r_val,tp,roff=roff,toff=toff)
            points = np.column_stack((R, T, np.zeros_like(R)))
            line = pv.MultipleLines(points)
            acteur = plotter.add_mesh(line,smooth_shading=True, color="black", line_width=2)
            plotter.renderer.RemoveActor(acteur) 
            static_renderer.AddActor(acteur)

        #photon sphere
        R,T = PD.penrose_coords(3.0,tp,roff=roff,toff=toff)
        line = pv.MultipleLines(np.column_stack((R, T, np.zeros_like(R))))
        acteur = plotter.add_mesh(line,smooth_shading=True, color="red", line_width=2)
        plotter.renderer.RemoveActor(acteur) 
        static_renderer.AddActor(acteur)

    @staticmethod
    def add_obj(obj, color="green"):
        actor = plotter.add_mesh(obj, color=color, line_width=2)
        plotter.renderer.RemoveActor(actor) 
        static_renderer.AddActor(actor)


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
plotter = pv.Plotter()

x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = potential(X, Y, init_mass)

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
static_renderer.SetViewport(0.60, 0.1, 1.60, 0.6)
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
positionLine = plotter.add_mesh(line, color="cyan", line_width=5)

PD.draw(t,r,tp,rp)

position = pv.Sphere(radius=0.05,center=(0,0,0))
diagram = plotter.add_mesh(position, color="yellow", line_width=2)
plotter.renderer.RemoveActor(diagram) 
static_renderer.AddActor(diagram)

static_renderer.ResetCamera()
plotter.camera_position = 'iso'
plotter.set_background('white')

#Updates every tick
def update_vectors(velo,center):
    photon.points += velo
    pos.points[1] = center
    velocity.points[0] = center

    norm = np.linalg.norm(velo) #safety first
    if norm > 1e-8:
        velocity.points[1] = center + velo * 3 / norm
    else:
        velocity.points[1] = center

def update_diagram(r,t):
    R,T = PD.penrose_coords(r,t,0,0)
    target = np.array([R[0],T[0],0.0],dtype=float)
    position.translate(target - position.center, inplace=True)

def update(move=True):
    global inward,end,tau, positions
    x,y,z = photon.center
    r = np.sqrt(x**2 + y**2)
    E_eff = 1.0 if not massive else E
    
    if move:
        if r <= 2 * M + 0.02: #Little extra so that the simulation stops in reasonable time
            print("Event Horizon Reached. DUN DUN DUUN")
            end = True
            return
        if (x > maxx or x < minx) or (y > maxx or y < minx):
            print("Particle escaped grid boundary")
            end = True
            return
        
        #Flip logic
        pot = ((1 - 2*M / r) * (L**2 / r**2)) if not massive else ((1 - 2*M / r) * (1 + L**2 / r**2))

        if inward and (pot >= E_eff**2 - 1e-6):
            inward = not inward
            print(f"Turning point reached at r = {r:.2f}")

        tau += d_tau
    
    xp, yp = calculate_new_pos(x,y,M,L,E,d_tau,massive,inward)
    velo = np.array([xp-x,yp-y,0])
    update_vectors(velo,photon.center)
    update_diagram(r,tau)

    #draw curvy line
    positions = np.vstack([positions, np.array([[xp,yp,z_center * z_stretch]])])
    updated = pv.lines_from_points(positions)
    positionLine.mapper.dataset.copy_from(updated)
    
    if not massive:
        #Calculate colour of photon (from an observer at infinity)
        lapse = max(1e-6, 1 - 2*M/r)                    #Ensure no infinite wavelengths
        wavelength = orig_wavelength / np.sqrt(lapse)
        photon_actor.prop.color = get_rgb(wavelength)
    else:
        photon_actor.prop.color = "Green"

#Sliders and callbacks
def reset_simulation():
    global L, E, photon, positions, tau, inward

    inward = np.cos(np.deg2rad(angle)) < 0
        
    photon.translate(init_pos - np.array(photon.center), inplace=True)
    positions = np.array([init_pos, init_pos], dtype=float)
    positionLine.mapper.dataset.copy_from(pv.lines_from_points(positions))

    L = ang_momentum(init_pos[0],init_pos[1],np.deg2rad(angle), M, v, massive)
    E = Energy(init_pos[0],init_pos[1], v, M, massive)

    tau = 0.0
    update(move=False)

def update_mass(value):
    global M
    M = value
    new_z = potential(X, Y, value)
    grid.points[:, 2] = new_z.ravel() * z_stretch

    new_sphere = pv.Sphere(radius=2*value, center=(0, 0, z_center * z_stretch))
    plotter.add_mesh(new_sphere, color="black", name="black hole")
    new_ring = pv.Disc(center=(0, 0, z_center * z_stretch), inner=3.0*value - 0.1, outer=3.0*value,c_res=70)
    plotter.add_mesh(new_ring, color="red",name='photon_sphere')
    reset_simulation()

def update_x(value):
    if not end:
        try: x_slider.GetRepresentation().SetValue(init_pos[0]); return #only allow changes if simulation not running
        except NameError: return
    init_pos[0] = value
    reset_simulation()

def update_y(value):
    if not end:
        try: y_slider.GetRepresentation().SetValue(init_pos[1]); return      #TOFIX: The init angle of the photon changes whenever it is translated
        except NameError: return
    init_pos[1] = value
    reset_simulation()

def update_angle(value):
    global angle
    if not end:
        try: angle_slider.GetRepresentation().SetValue(angle); return #only allow changes if simulation not running
        except NameError: return
    angle = value % 360
    reset_simulation()    

def update_massive(value):
    global massive
    if not end:
        try: massive_button.GetRepresentation().SetState((not value)); return #only allow changes if simulation not running
        except NameError: return
    massive = value
    reset_simulation()

def update_velo(value):
    global v
    if not massive or not end:
        try: velo_slider.GetRepresentation().SetValue(v) #only allow changes if simulation not running
        except NameError: pass
        return
    v = value
    reset_simulation()

def reset_view():
    update_mass(init_mass)
    try: mass_slider.GetRepresentation().SetValue(init_mass)
    except NameError: pass
    
def start_simulation(value):
    global end
    if value:
        end = False
        reset_simulation()
        for i in range(steps):
            if end: break
            update()
            plotter.update()
        end = True
        reset_simulation()
        try: start_button.GetRepresentation().SetState(0)
        except NameError: pass
    else:
        end = True
        reset_simulation()
#UI

mass_slider  = plotter.add_slider_widget(callback=update_mass, rng=[0.1, 4.0], value=init_mass, title="Mass", pointa=(0.02, 0.1), pointb=(0.24, 0.1), style='modern', color='black')
x_slider     = plotter.add_slider_widget(callback=update_x, rng=[minx, maxx], value=init_pos[0], title="Initial X", pointa=(0.26, 0.1), pointb=(0.48, 0.1), style='modern', color='black')
y_slider     = plotter.add_slider_widget(callback=update_y, rng=[minx, maxx], value=init_pos[1], title="Initial Y", pointa=(0.50, 0.1), pointb=(0.72, 0.1), style='modern', color='black')
angle_slider = plotter.add_slider_widget(callback=update_angle, rng=[0, 360], value=angle, title="Initial Direction", pointa=(0.74, 0.1), pointb=(0.96, 0.1), style='modern', color='black')
velo_slider  = plotter.add_slider_widget(callback=update_velo, rng=[0.05, 0.999], value=v, title="Velocity", pointa=(0.74, 0.8), pointb=(0.96, 0.8), style='modern', color='black')

start_button = plotter.add_checkbox_button_widget(start_simulation, value=False, position=(20, 150), size=button_size, color_on='green', color_off='grey')
plotter.add_text("Start/Reset Simulation", position=(20 + button_size + 10, 152), font_size=10, color="black", name="button_label")

massive_button = plotter.add_checkbox_button_widget(update_massive, value=True, position=(plotter.window_size[0] - 95, plotter.window_size[1] - 120), size=button_size, color_on='Purple', color_off='Green')
plotter.add_text("Toggle photon/massive     ", position='upper_right', font_size=10, color="black", name="button2_label")

plotter.add_text("Press 'R' to reset view", position='upper_left', color='black', font_size=10)
plotter.add_key_event('r', reset_view)

reset_simulation()

plotter.ren_win.SetFullScreen(True)
plotter.show(interactive_update=True,full_screen=True)


    
