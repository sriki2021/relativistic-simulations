import pyvista as pv
import numpy as np
import warnings
import time
import colour #pip install colour-science
import vtk

# Suppress warnings for invalid math inside the event horizon
#warnings.filterwarnings("ignore", category=RuntimeWarning)

def potential(x, y, m):
    """Calculate the Schwarzschild effective potential."""
    r = np.sqrt(x**2 + y**2)
    v_sq = 1 - 2 * m / r
    v_sq[v_sq < 0] = np.nan
    return np.sqrt(v_sq)
##def ang_momentum(x, y, vx, vy, m):
##    r = np.sqrt(x**2 + y**2)
##    v_mag = np.sqrt(vx**2 + vy**2)
##    ux, uy = vx/v_mag, vy/v_mag
##
##    r_cross_v = abs(x * uy - y * ux) # cross product
##    return r_cross_v / np.sqrt(1 - 2 * m / r)

def ang_momentum(x, y, alpha, m):
    r = np.sqrt(x**2 + y**2)
    return r * np.sin(alpha) / np.sqrt(1 - 2 * m / r)

def calc_rad_velocity(r,L,M,inward):
    E = 1
    rad_v = np.sqrt(np.clip((E**2 - (1 - 2*M / r) * (L**2 / r**2)),0,None))
    if inward:
        return -1 * rad_v
    return rad_v
def calc_ang_velocity(r,L):
    return L / r**2

def calculate_new_pos(x,y,M,L,lamb,inward=True):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    #k1
    rad_v = calc_rad_velocity(r,L,M,inward)
    ang_v = calc_ang_velocity(r,L)

    r2 = r + rad_v * lamb/2
    p2 = phi + ang_v * lamb/2

    #k2
    rad_v2 = calc_rad_velocity(r2,L,M,inward)
    ang_v2 = calc_ang_velocity(r2,L)

    r3 = r + rad_v2 * lamb/2

    p3 = phi + ang_v2 * lamb/2

    #k3
    rad_v3 = calc_rad_velocity(r3,L,M,inward)
    ang_v3 = calc_ang_velocity(r3,L)

    r4 = r + rad_v3 * lamb
    p4 = phi + ang_v3 * lamb

    #k4
    rad_v4 = calc_rad_velocity(r4,L,M,inward)
    ang_v4 = calc_ang_velocity(r4,L)

    r_final = r + lamb/6 * (rad_v + 2 * rad_v2 + 2 * rad_v3 + rad_v4)
    phi_final = phi + lamb/6 * (ang_v + 2 * ang_v2 + 2 * ang_v3 + ang_v4)

    x_final = r_final * np.cos(phi_final)
    y_final = r_final * np.sin(phi_final)
    return (x_final,y_final)

#Helper function to obtain RGB color from a wavelength
def get_rgb(nm):
    rgb = colour.convert(max(360,min(780,nm)), 'Wavelength', 'sRGB')
    return tuple(rgb.clip(0, 1))

#Parameters
minz, maxz = 0.90, 1.07
z_center = (maxz + minz) / 2
init_mass = 1.0
z_stretch = 20.0
lambada = 0.01 #default accuracy
orig_wavelength = 450
steps = 20000
M = init_mass
button_size = 30


#Changables
minx, maxx = -10, 10
init_dir = 135 #Angle measured relative to the radial v 
init_pos = np.array([5,5 , z_center * z_stretch])
velo = np.array([0,0,0])

prev_values = np.array([init_pos[0],init_pos[1],init_dir])
inward = True
end = True

#Setting up the potential mesh-grid
x = np.linspace(minx, maxx, 60)
y = np.linspace(minx, maxx, 60)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = potential(X, Y, init_mass)
L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * init_dir,init_mass)

grid = pv.StructuredGrid(X, Y, Z * z_stretch)

x2 = np.linspace(-5, 5, 5)
y2 = np.linspace(-5, 5, 5)
X2, Y2 = np.meshgrid(x2, y2, indexing='ij')
Z2 = np.zeros_like(X2)
penrose = pv.StructuredGrid(X2, Y2, Z2)


black = pv.Sphere(radius=2.0, center=(0, 0, z_center * z_stretch))
photon_sphere = pv.Disc(center=(0, 0, z_center * z_stretch), inner=2.9, outer=3.0,c_res=70)

photon = pv.Sphere(radius=0.1, center=init_pos)
pos = pv.Line((0, 0, z_center * z_stretch), init_pos)
velocity = pv.Line(init_pos, init_pos)

plotter = pv.Plotter()

#2nd renderer for diagram
static_renderer = vtk.vtkRenderer()
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

diagram = plotter.add_mesh(penrose, style='wireframe', color='yellow', line_width=1, name='penrose')
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

def update():
    global inward,end
    x,y,z = photon.center
    r = np.sqrt(x**2 + y**2)
    dynamic_lamb = lambada * (r / (3 * M)) #dynamic oooo    

    if r <= 2 * M:
        print("GOTem Photon fell into the event horizon.")
        end = True
    if (x > maxx or x < minx) or (y > maxx or y < minx):
        print("Photon vanished")
        end = True
    
    pot = (1 - 2*M / r) * (L**2 / r**2)
    if inward and (pot >= 1.0):
        inward = False
        print(f"T.p. reached at r = {r:.2f}")
    
    xp, yp = calculate_new_pos(x,y,M,L,lambada,inward)
    velo = np.array([xp-x,yp-y,0])
    update_vectors(velo,photon.center)

    #calculate colour of photon (from an observer at infinity)
    wavelength = orig_wavelength / np.sqrt(1 - 2*M / r)
    photon_actor.prop.color = get_rgb(wavelength)

    time_dilation = 0.01/ np.sqrt(1 - 2*M / r) #time dilation
    if(not np.isnan(time_dilation)):
        time.sleep(time_dilation)
    


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
    global prev_values
    if not end:
        try: x_slider.GetRepresentation().SetValue(prev_values[0]) #only allow changes if simulation not running
        except NameError: pass
        return
    prev_values[0] = value
    init_pos[0] = value
    photon.translate((value - photon.center[0],0,0), inplace=True)
    update()
    try:
        L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * angle_slider.GetRepresentation().GetValue(),init_mass)
        plotter.update()
    except (RuntimeError, NameError):
        pass
    return

def update_y(value):
    global prev_values
    if not end:
        try: y_slider.GetRepresentation().SetValue(prev_values[1])      #TOFIX: The init angle of the photon changes whenever it is translated
        except NameError: pass
        return
    prev_values[1] = value
    init_pos[1] = value
    photon.translate((0,value - photon.center[1],0), inplace=True)
    
    update()
    try:
        plotter.update()
    except RuntimeError:
        pass
    return

def update_angle(value):
    global L, prev_values,inward
    if not end:
        try: angle_slider.GetRepresentation().SetValue(prev_values[2]) #only allow changes if simulation not running
        except NameError: pass
        return
    prev_values[2] = value
    if (0 <= value < 90 or 270 < value <= 360):
        inward = False
    else:
        inward = True
    #recalculate L
    L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * value,init_mass)
    update()
    try:
        plotter.update()
    except RuntimeError:
        pass
    return

def reset_view():
    update_mass(init_mass)
    plotter.slider_widgets[0].GetRepresentation().SetValue(init_mass)

def reset_simulation():
    global L, photon
    L = ang_momentum(init_pos[0],init_pos[1],np.pi / 180 * angle_slider.GetRepresentation().GetValue(),init_mass) # okay works?
    photon.translate(init_pos - np.array(photon.center), inplace=True)
    
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

plotter.add_text("Press 'R' to reset physics view", position='upper_left', color='black', font_size=10)
plotter.add_key_event('r', reset_view)



plotter.show(interactive_update=True)


    
