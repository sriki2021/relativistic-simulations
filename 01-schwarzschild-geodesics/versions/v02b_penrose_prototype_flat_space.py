import pyvista as pv
import vtk
import numpy as np
import time

plotter = pv.Plotter()

#Second renderer for penrose diagram (so that it isn't altered by camera view
static_renderer = vtk.vtkRenderer()
static_renderer.SetViewport(0.60, 0.1, 1.60, 0.6)
static_renderer.SetLayer(1)
static_renderer.InteractiveOff()

plotter.ren_win.SetNumberOfLayers(2)
plotter.ren_win.AddRenderer(static_renderer)

#zigzag coords
res = 15
n_points = 40
amplitude = 0.05
width = 2

#t and r coords, rp and tp are more precise values to draw each curve accurately
t = np.linspace(-np.pi/2.,np.pi/2,res,endpoint=False)[1:]
tp = np.linspace(-np.pi/2.,np.pi/2,50,endpoint=False)[1:]
r = np.linspace(-np.pi/2.,np.pi/2,res,endpoint=False)[1:]
rp = np.linspace(-np.pi/2.,np.pi/2,50,endpoint=False)[1:]

#Coords for the edges of the diagram
top    = [0,  1, 0]
bottom = [0, -1, 0]
right  = [1, 0,  0]
left   = [-1, 0, 0]
leftest =[-2, 1, 0]

#Zigzag at singularity
sx = np.linspace(-2, 0, n_points)
sy = np.zeros(n_points)
sy[::2] = 1+amplitude/2
sy[1::2] = 1-amplitude/2
sz = np.zeros(n_points)
spoints = np.column_stack((sx, sy, sz))


class PD:#Penrose Diagram

    #Gives the penrose coordinates (plus an offset)
    def penrose_coords(r,t,roff=0,toff=0):
        R = 1./np.pi*(np.arctan(t+r)-np.arctan(t-r))
        T = 1./np.pi*(np.arctan(t+r)+np.arctan(t-r))
        return R+roff,T+toff

    #Draws a diagram given t and r (Range of vals can be chosen) offset also available
    @staticmethod
    def draw_diagram(t,r,tp,rp,roff=0,toff=0,mint=-np.pi/2,maxt=np.pi/2,minr=-np.pi/2,maxr=np.pi/2):
        t = t[(t >= mint) & (t <= maxt)]
        tp = tp[(tp >= mint) & (tp <= maxt)]
        r = r[(r >= minr) & (r <= maxr)]
        rp = rp[(rp >= minr) & (rp <= maxr)]
        for t_val in t:
            R,T = PD.penrose_coords(np.tan(rp),np.tan(t_val),roff=roff,toff=toff)
            points = np.column_stack((R, T, np.zeros_like(R))) 
            line = pv.MultipleLines(points)
            acteur = plotter.add_mesh(line,smooth_shading=True, color="black", line_width=2)
            
            plotter.renderer.RemoveActor(acteur) 
            static_renderer.AddActor(acteur)

        for r_val in r:
            R,T = PD.penrose_coords(np.tan(r_val),np.tan(tp),roff=roff,toff=toff)
            points = np.column_stack((R, T, np.zeros_like(R)))
            line = pv.MultipleLines(points)
            acteur = plotter.add_mesh(line, color="black", line_width=2)
            plotter.renderer.RemoveActor(acteur) 
            static_renderer.AddActor(acteur)

    def add_obj(obj, color="green"):
        actor = plotter.add_mesh(obj, color=color, line_width=2)
        plotter.renderer.RemoveActor(actor) 
        static_renderer.AddActor(actor)


    def draw(t,r,tp,rp):
        #QUADRANT 1
        PD.draw_diagram(t,r,tp,rp,roff=0,toff=0)

        #QUADRANT 2
        PD.draw_diagram(t,r,tp,rp,roff=-1,toff=1,maxt=0)

        static_renderer.ResetCamera()
        boundary_points = np.array([bottom, right, top, left, bottom])
        triangle = np.array([top, left, leftest])
        lines1 = pv.MultipleLines(points=boundary_points)
        lines2 = pv.MultipleLines(points=triangle)
        zigzag = pv.MultipleLines(spoints)

        PD.add_obj(lines1,color="green")
        PD.add_obj(lines2,color="green")
        PD.add_obj(zigzag,color="red")

PD.draw(t,r,tp,rp)


init_pos = np.array([0.0,0.0,0.0])
change = np.array([0.1,0.0,0.0])


position = pv.Sphere(radius=0.05,center=(0,0,0))
diagram = plotter.add_mesh(position, color="yellow", line_width=2)
plotter.renderer.RemoveActor(diagram)
static_renderer.AddActor(diagram)
def starter():
    t=0
    current = np.array([0.0,0.0,0.0])
    for i in range(1000):
        current += change
        t += 0.1
        r = np.dot(current-init_pos,current-init_pos)
        R,T = PD.penrose_coords(r,t)
        r0,t0,z = position.center
        print(r,t)
    #shift = [r - current_center[0], t - current_center[1], 0]
        position.points += [R-r0,T-t0,0]
        plotter.update()
        time.sleep(0.1)
        
        
    print("done")

    
plotter.add_key_event('r', starter)
plotter.show(interactive_update=True)












