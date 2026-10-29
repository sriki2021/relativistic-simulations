import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from matplotlib.widgets import Button, Slider
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

#field function
def f(x, y, m):
    r = np.sqrt(x**2 + y**2)
    #L = 3.9
    L = 0
    M = m
    # Effective Potential = Vsq = (1 - 2*M/r) * (1+(L**2)/(r**2))
    Vsq = (1 - 2*M/r)
    V = np.sqrt(Vsq)
    return V

#vector functions
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

minz, maxz = 0.90, 1.07
minx, maxx = -100, 100
init_mass = 1

x = np.linspace(minx, maxx, 1000)
y = np.linspace(minx, maxx, 1000)
X, Y = np.meshgrid(x, y)
Z = f(X, Y, init_mass)

#draw potential field
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
field = ax.plot_wireframe(X, Y, Z, color='black', rstride=40, cstride=40)

#draw black hole centre
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = 2 * np.outer(np.cos(u), np.sin(v))
y = 2 * np.outer(np.sin(u), np.sin(v))
k = (maxz - minz)/(maxx - minx)
z = (maxz + minz)/2 + 2 * k * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='black', shade=False, antialiased=False)


#draw photon
uP = np.linspace(0, 2 * np.pi, 20)
vP = np.linspace(0, np.pi, 20)
xp = 50 + 1 * np.outer(np.cos(uP), np.sin(vP))
yp = 50 + 1 * np.outer(np.sin(uP), np.sin(vP))
zp = (maxz + minz)/2 + 1 * k * np.outer(np.ones(np.size(uP)), np.cos(vP))
ax.plot_surface(xp, yp, zp, color='red', shade=False, antialiased=False)

ax.quiver(0, 0, zp, 50, 50, 0, 
          arrow_length_ratio=0, 
          linewidths=2, 
          capstyle='round')



#coloring?s
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)


ax.set_title('something')
ax.set_box_aspect((1, 1, 1),zoom=1.4)

ax.set_zlim(minz,maxz)
ax.set_xlim(minx,maxx)
ax.set_ylim(minx,maxx)
ax.view_init(elev=30, azim=45, roll=0, vertical_axis='z')

#ax.disable_mouse_rotation()
ax.set_axis_off() 
#fig.canvas.manager.full_screen_toggle()
#plt.tight_layout(pad=0)
fig.subplots_adjust(left=0.25, bottom=0)

axmass = fig.add_axes([0.04, 0.75, 0.230, 0.03])
mass_slider = Slider(
    ax=axmass,
    label='Mass',
    valmin=0.1,
    valmax=3,
    valinit=init_mass,
    orientation="horizontal"
)

#update the field whenever the mass slider changes
def update(val):
    global field
    m = mass_slider.val
    field.remove()
    Z_new = f(X, Y, m)
    field = ax.plot_wireframe(X, Y, Z_new, color='black', rstride=40, cstride=40)
    fig.canvas.draw_idle()

mass_slider.on_changed(update)

resetax = fig.add_axes([0.05, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    mass_slider.reset()
button.on_clicked(reset)

plt.show()
