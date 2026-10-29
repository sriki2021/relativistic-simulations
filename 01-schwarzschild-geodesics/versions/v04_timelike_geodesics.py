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

lambada = 0.01 #default accuracy
time= 0.0
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
velo = np.array([0,0,0])
penrose_scale = 1

massive=True
E=1
v=0.9

prev_values = np.array([init_pos[0],init_pos[1],init_dir,v])
positions = np.array([init_pos, init_pos], dtype=float)
line = pv.lines_from_points(positions)
inward = True
end = True

def potential(x, y, m):
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 * m / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)
def gamma(v):
    return 1 / np.sqrt(1 - v**2)

def Energy(x, y, v, M): #specific energy
    if massive:
        r = np.sqrt(x**2 + y**2)
        return np.sqrt(1 - 2*M/r) * gamma(v)
    return 1

def rad_event(r):
    global E,L,M
    if massive:
        return E**2 - (1 - 2*M / r)*(1 + L**2 / r**2)
    return E**2 - (1 - 2*M / r)*(L**2 / r**2)
def ang_momentum(x, y, alpha, M): #specific ang momentum
    global v
    r = np.sqrt(x**2 + y**2)
    if massive:
        return r * np.sin(alpha) * gamma(v) * v
    return r * np.sin(alpha) / np.sqrt(1 - 2 * M / r)

def calc_rad_velocity(r,L,M,inward):
    global E
    if not massive:
        E_ef = 1.0
        b = L / E_ef
        rad_v = (1 - 2*M/r) * np.sqrt(np.clip((1 - (1 - 2*M / r) * (b**2 / r**2)),0,None))
    else:
        rad_v = (1 - 2*M/r) * np.sqrt(np.clip((E**2 - (1 - 2*M / r) * (1 + L**2 / r**2)),0,None)) / E
    return -rad_v if inward else rad_v

def calc_ang_velocity(r,L,M):
    global E
    E_ef = 1.0 if not massive else E
    return (1 - 2*M / r) * L / (E_ef * r**2)

def calculate_new_pos(x,y,M,L,lamb,inward=True): # RK4
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    #k1
    rad_v = calc_rad_velocity(r,L,M,inward)
    ang_v = calc_ang_velocity(r,L,M)

    r2 = r + rad_v * lamb/2
    p2 = phi + ang_v * lamb/2

    #k2
    rad_v2 = calc_rad_velocity(r2,L,M,inward)
    ang_v2 = calc_ang_velocity(r2,L,M)

    r3 = r + rad_v2 * lamb/2

    p3 = phi + ang_v2 * lamb/2

    #k3
    rad_v3 = calc_rad_velocity(r3,L,M,inward)
    ang_v3 = calc_ang_velocity(r3,L,M)

    r4 = r + rad_v3 * lamb
    p4 = phi + ang_v3 * lamb

    #k4
    rad_v4 = calc_rad_velocity(r4,L,M,inward)
    ang_v4 = calc_ang_velocity(r4,L,M)

    r_final = r + lamb/6 * (rad_v + 2 * rad_v2 + 2 * rad_v3 + rad_v4)
    phi_final = phi + lamb/6 * (ang_v + 2 * ang_v2 + 2 * ang_v3 + ang_v4)

    x_final = r_final * np.cos(phi_final)
    y_final = r_final * np.sin(phi_final)
    return (x_final,y_final)

#Helper function to obtain RGB color from a wavelength
def get_rgb(nm):
    rgb = colour.convert(max(360,min(780,nm)), 'Wavelength', 'sRGB')
    return tuple(rgb.clip(0, 1))


class PD:#Penrose Diagram

    #Gives the penrose coordinates (plus an offset)
    def penrose_coords(r_in,t_in,roff=0,toff=0):
        #R = 1./np.pi*(np.arctan(t+r)-np.arctan(t-r))       #Old for flat space-time
        #T = 1./np.pi*(np.arctan(t+r)+np.arctan(t-r))

        r = np.atleast_1d(np.array(r_in,dtype="float"))
        t = np.atleast_1d(np.array(t_in,dtype="float"))

        if t.size == 1 and r.size>1:
            t = np.full_like(r,t[0])
        if r.size == 1 and t.size>1:
            r = np.full_like(t,r[0])

        U = np.zeros_like(r)
        V = np.zeros_like(r)

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
        
        
        #r = np.where(r > 2 * M, np.sqrt(r / (2 * M) - 1) * np.exp(r /(4 * M)) * np.cosh(t / (4 * M)), np.sqrt(1 - r / (2 * M)) * np.exp(r /(4 * M)) * np.sinh(t / (4 * M)))
        #t = np.where(r > 2 * M, np.sqrt(r / (2 * M) - 1) * np.exp(r /(4 * M)) * np.sinh(t / (4 * M)), np.sqrt(1 - r / (2 * M)) * np.exp(r /(4 * M)) * np.cosh(t / (4 * M)))
        
        #T = np.sqrt(r / (2 * M) - 1) * np.exp(r /(4 * M)) * np.sinh(t / (4 * M))
##        return R+roff,T+toff
##        R = np.sqrt(1 - r / (2 * M)) * np.exp(r /(4 * M)) * np.sinh(t / (4 * M))
##        T = np.sqrt(1 - r / (2 * M)) * np.exp(r /(4 * M)) * np.cosh(t / (4 * M))  #inside the black hole

        VplusU = (V + U) / penrose_scale
        VminusU = (V - U) / penrose_scale
        
        Tp = 2./np.pi*(np.arctan(VplusU) + np.arctan(VminusU))
        Rp = 2./np.pi*(np.arctan(VplusU) - np.arctan(VminusU))
        return Rp+roff,Tp+toff

    #Draws a diagram given t and r (Range of vals can be chosen) offset also available
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
        points = np.column_stack((R, T, np.zeros_like(R)))
        line = pv.MultipleLines(points)
        acteur = plotter.add_mesh(line,smooth_shading=True, color="red", line_width=2)
        plotter.renderer.RemoveActor(acteur) 
        static_renderer.AddActor(acteur)

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
x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = potential(X, Y, init_mass)
L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * init_dir,M)
E = Energy(init_pos[0],init_pos[1], v, M)
grid = pv.StructuredGrid(X, Y, Z * z_stretch)
black = pv.Sphere(radius=2.0, center=(0, 0, z_center * z_stretch))
photon_sphere = pv.Disc(center=(0, 0, z_center * z_stretch), inner=2.9, outer=3.0,c_res=70)
angle = init_dir

photon = pv.Sphere(radius=0.1, center=init_pos)
pos = pv.Line((0, 0, z_center * z_stretch), init_pos)
velocity = pv.Line(init_pos, init_pos)

plotter = pv.Plotter()

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

##rgb_list = [get_rgb(w) for w in range(380, 751)]
##cmap = LinearSegmentedColormap.from_list("spectrum", custom_rgb_list) #colour map for wavelength?
##
##pl.add_mesh(mesh, scalars="Wavelength", cmap=cmap)

plotter.camera_position = 'iso'
plotter.set_background('white')

def update_vectors(velo,center):
    photon.points += velo
    pos.points[1] = center
    velocity.points[0] = center
    velocity.points[1] = center + velo * 3 / np.linalg.norm(velo)
    

def update_diagram(r,t):

    R,T = PD.penrose_coords(r,t,0,0)
    target = np.array([R[0],T[0],0.0],dtype=float)
    shift =  target - position.center
    position.translate(shift, inplace=True)

def update(move=True):
    global inward,end,time, positions
    x,y,z = photon.center
    r = np.sqrt(x**2 + y**2)
    #dynamic_lamb = lambada / (1 - 2 / r) #dynamic oooo
    
    if move:
        if r <= 2 * M + 0.02: #extra outside to avoid prolonged wait
            print("GOTem Photon fell into the event horizon.")
            end = True
            return
        if (x > maxx or x < minx) or (y > maxx or y < minx):
            print("Photon vanished")
            end = True
            return
        #flip logic
        pot = ((1 - 2*M / r) * (L**2 / r**2)) if not massive else ((1 - 2*M / r) * (1 + L**2 / r**2))
        if inward and (pot >= E**2 - 1e-6):
            inward = not inward
            print(f"T.p. reached at r = {r:.2f}")

        time += lambada
    
    xp, yp = calculate_new_pos(x,y,M,L,lambada,inward) #not lambada
    velo = np.array([xp-x,yp-y,0])
    update_vectors(velo,photon.center)
    update_diagram(r,time)

    #draw curvy line
    #print(f"Position shape: {position.shape}, New point shape: {np.array([[xp,yp,0]]).shape}")
    positions = np.vstack([positions, np.array([[xp,yp,z_center * z_stretch]])])
    updated = pv.lines_from_points(positions)
    positionLine.mapper.dataset.copy_from(updated)
    if not massive:
        #calculate colour of photon (from an observer at infinity)
        lapse = max(1e-6, 1 - 2*M/r) #make sure no infinite wavelengths
        wavelength = orig_wavelength / np.sqrt(lapse)
        photon_actor.prop.color = get_rgb(wavelength)
    else:
        photon_actor.prop.color = "Green"


def update_mass(value):
    global M
    M = value
    new_z = potential(X, Y, value)
    grid.points[:, 2] = new_z.ravel() * z_stretch

    new_sphere = pv.Sphere(radius=2*value, center=(0, 0, z_center * z_stretch))
    plotter.add_mesh(new_sphere, color="black", name="black hole")

    new_ring = pv.Disc(center=(0, 0, z_center * z_stretch), inner=3.0*value - 0.1, outer=3.0*value,c_res=70)
    plotter.add_mesh(new_ring, color="red",name='photon_sphere')

def update_x(value):
    global prev_values,M,E
    if not end:
        try: x_slider.GetRepresentation().SetValue(prev_values[0]) #only allow changes if simulation not running
        except NameError: pass
        return
    prev_values[0] = value
    init_pos[0] = value
    photon.translate((value - photon.center[0],0,0), inplace=True)
    
    update(move=False)
    try:
        L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * angle,M)
        E = Energy(init_pos[0],init_pos[1], v, M)
        plotter.update()
    except (RuntimeError, NameError):
        pass
    reset_simulation()
    return

def update_y(value):
    global prev_values,M,E
    if not end:
        try: y_slider.GetRepresentation().SetValue(prev_values[1])      #TOFIX: The init angle of the photon changes whenever it is translated
        except NameError: pass
        return
    prev_values[1] = value
    init_pos[1] = value
    photon.translate((0,value - photon.center[1],0), inplace=True)
    
    update(move=False)
    try:
        L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * angle,M)
        E = Energy(init_pos[0],init_pos[1], v, M)
        plotter.update()
    except (RuntimeError, NameError):
        pass
    reset_simulation()
    return

def update_angle(value):
    global L, prev_values,inward,angle,M,E
    if not end:
        try: angle_slider.GetRepresentation().SetValue(prev_values[2]) #only allow changes if simulation not running
        except NameError: pass
        return
    prev_values[2] = value
    angle = value % 360
    alpha = np.deg2rad(value)
    if (0 <= value < 90 or 270 < value <= 360):
        inward = False
    else:
        inward = True
    #recalculate L
    print(angle)
##    if massive:
##        ur = gamma(v) * v * np.cos(alpha)
##        uphi = gamma(v) * v * np.sin(alpha)
##        vr = np.sqrt(1 - 2M/r)*ur
##        vphi = 1/r * uphi
##        L = r**2 * vphi
    


    
    
    L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * angle,M)
    update(move=False)
    try:
        plotter.update()
    except RuntimeError:
        pass
    reset_simulation()
    return

def update_massive(value):
    global massive,end,E,L
    if not end:
        try: massive_button.GetRepresentation().SetState((not value)) #only allow changes if simulation not running
        except NameError: pass
        return
    E = Energy(init_pos[0], init_pos[1], v, M)
    L = ang_momentum(init_pos[0], init_pos[1], np.deg2rad(angle), M)
    massive = value
    reset_simulation()

def update_velo(value):
    global v
    if not massive or not end:
        try: velo_slider.GetRepresentation().SetValue(prev_values[3]) #only allow changes if simulation not running
        except NameError: pass
        return
    prev_values[3] = value
    v = value

def reset_view():
    update_mass(init_mass)
    plotter.slider_widgets[0].GetRepresentation().SetValue(init_mass)

def angle_increment():
    global angle
    angle += 2
    angle = angle % 360
    print(angle)
    reset_simulation()
    
def reset_simulation():
    global L, photon, positions,time,M,init_pos,v,angle,E
    
    photon.translate(init_pos - np.array(photon.center), inplace=True)
    positions = np.array([init_pos, init_pos], dtype=float)
    reset_line = pv.lines_from_points(positions)
    positionLine.mapper.dataset.copy_from(reset_line)
    L = ang_momentum(init_pos[0],init_pos[1],np.deg2rad(angle),M)
    E = Energy(init_pos[0],init_pos[1], v, M)
    time = 0.0
    update(move=False)
        
        

    #except NameError: pass
    time = 0.0
    
    
def start_simulation(value):
    global end
    if value:
        print("start")
        end = False
        main_loop()
        start_button.GetRepresentation().SetState(0)
    else:
        end = True
        

def main_loop(): #ban recursive calls?
    global end
    reset_simulation()
    for i in range(steps):
        if end:
            break
        #try:
        update()
        plotter.update()
        #except (RuntimeError, AttributeError):
        #print("Window closed.")
            #break
    end = True
    #reset_simulation()
    #`update()

#x_slider
#y_slider
#angle_slider
# Replace Matplotlib Slider
mass_slider = plotter.add_slider_widget(
    callback=update_mass,
    rng=[0.1, 4.0],
    value=init_mass,
    title="Mass",
    pointa=(0.02, 0.1),
    pointb=(0.24, 0.1),
    style='modern',
    color='black'
)
# Add x,y and direction sliders
x_slider = plotter.add_slider_widget(
    callback=update_x,
    rng=[minx, maxx],
    value=prev_values[0],
    title="Initial X",
    pointa=(0.26, 0.1),
    pointb=(0.48, 0.1),
    style='modern',
    color='black'
)
y_slider = plotter.add_slider_widget(
    callback=update_y,
    rng=[minx, maxx],
    value=prev_values[1],
    title="Initial Y",
    pointa=(0.50, 0.1),
    pointb=(0.72, 0.1),
    style='modern',
    color='black'
)
angle_slider = plotter.add_slider_widget(
    callback=update_angle,
    rng=[0, 360],
    value=prev_values[2],
    title="Initial Direction",
    pointa=(0.74, 0.1),
    pointb=(0.96, 0.1),
    style='modern',
    color='black'
)


start_button = plotter.add_checkbox_button_widget(
    start_simulation, 
    value=False, 
    position=(20, 150), 
    size=button_size,
    color_on='green',
    color_off='grey'
)

plotter.add_text(
    "Start/Reset Simulation", 
    position=(20 + button_size + 10, 152), 
    font_size=10, 
    color="black",
    name="button_label"
)

massive_button = plotter.add_checkbox_button_widget(
    update_massive, 
    value=True, 
    position=(plotter.window_size[0] - 95, plotter.window_size[1] - 120), 
    size=button_size,
    color_on='Purple',
    color_off='Green'
)

plotter.add_text(
    "\nToggle photon/massive     ", 
    #position=(plotter.window_size[0] - 150 + button_size + 10, plotter.window_size[1] - 98),
    position='upper_right',
    font_size=10, 
    color="black",
    name="button2_label"
)

velo_slider = plotter.add_slider_widget(
    callback=update_velo,
    rng=[0.05, 0.999],
    value=prev_values[3],
    title="Velocity",
    pointa=(0.74, 0.8),
    pointb=(0.96, 0.8),
    style='modern',
    color='black'
)


plotter.add_text("Press 'R' to reset physics view", position='upper_left', color='black', font_size=10)
plotter.add_key_event('r', reset_view)

plotter.add_key_event('a', angle_increment)

plotter.ren_win.SetFullScreen(True)
plotter.show(interactive_update=True,full_screen=True)


    
