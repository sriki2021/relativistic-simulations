"""
Driver program - runs every measurement needed for the five figures

    python run_all.py             everything
    python run_all.py shadow      just one group: shadow | drift | timing
"""
import os, sys, subprocess, time

GROUP = sys.argv[1] if len(sys.argv) > 1 else "all"
PY = sys.executable


def run(**env):
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    t0 = time.perf_counter()
    subprocess.run([PY, "measure.py"], env=e, check=True)
    print(f"    ({time.perf_counter()-t0:.0f}s)\n")


# Figures 1 and 5: shadow boundary

if GROUP in ("all", "shadow"):
    SPINS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.998]
    for i, spin in enumerate(SPINS):
        print(f"--- shadow {i+1}/{len(SPINS)}  a={spin}")

        run(BH_MODE="shadow", BH_A=spin,
            BH_WIDTH=256, BH_HEIGHT=1, BH_DISK=0,
            BH_TOL=1e-8, BH_STEPS=200000, BH_RADIUS=50, BH_CLAMP=0)


# Figures 2 and 3: first-integral drift, RK45 vs RK4
if GROUP in ("all", "drift"):

    COMMON = dict(BH_MODE="drift", BH_A=0.5, BH_WIDTH=128, BH_HEIGHT=1,
                  BH_DISK=0, BH_STEPS=200000, BH_RADIUS=50, BH_CLAMP=0)

    for t in ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]:
        print(f"--- drift rk45 tol={t}")
        run(BH_INTEG="rk45", BH_TOL=t, **COMMON)

    for h in ["3e-2", "1e-2", "3e-3", "1e-3", "3e-4", "1e-4"]:
        print(f"--- drift rk4  h={h}")
        run(BH_INTEG="rk4", BH_H=h, BH_TOL="1e-6", **COMMON)


# Figure 3: cost to render the photon ring to a required accuracy
if GROUP in ("all", "cost"):
    COMMON = dict(BH_MODE="cost", BH_A=0.5, BH_WIDTH=64, BH_HEIGHT=1,
                  BH_DISK=0, BH_STEPS=400000, BH_RADIUS=50, BH_CLAMP=0)
    for t in ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]:
        print(f"--- cost rk45 tol={t}")
        run(BH_INTEG="rk45", BH_TOL=t, **COMMON)
    for h in ["1e-2", "3e-3", "1e-3", "3e-4"]:
        print(f"--- cost rk4  h={h}")
        run(BH_INTEG="rk4", BH_H=h, BH_TOL="1e-6", **COMMON)


# Figure 4: CPU vs GPU wall-clock
if GROUP in ("all", "timing"):
    RES = [(320, 180), (480, 270), (640, 360), (960, 540)]
    for arch in ["cuda", "cpu"]:
        for (w, h) in RES:
            if arch == "cpu" and w > 640:
                continue                      # skip the slowest CPU points
            print(f"--- timing {arch} {w}x{h}")
            run(BH_MODE="timing", BH_ARCH=arch, BH_WIDTH=w, BH_HEIGHT=h,
                BH_A=0.5, BH_TOL=1e-6, BH_DISK=1, BH_REPS=5)

print("done. now: python make_figures.py")
