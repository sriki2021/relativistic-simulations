
import os, time
import numpy as np
import taichi as ti
import main_v16 as bh

os.makedirs("data", exist_ok=True)
MODE = os.environ["BH_MODE"]

CAM  = ti.Vector([0.0, 15.0, 0.0])
LOOK = ti.Vector([0.0, -15.0, 0.0])

CAM_D  = ti.Vector([0.0,  15.0,  6.0])
LOOK_D = ti.Vector([0.0, -15.0, -6.0])


# ---------------------------------------------------------------- shadow ----
if MODE == "shadow":
    psis = np.linspace(0, 2*np.pi, bh.width, endpoint=False)
    xi, eta, bracket = bh.shadow_boundary(CAM, LOOK, psis)
    np.savez(f"data/shadow_a{bh.a:.4f}.npz",
             a=bh.a, psis=psis, xi=xi, eta=eta, bracket=bracket)
    print(f"[shadow] a={bh.a:.4f}  b_crit range "
          f"{np.sqrt(xi**2+eta).min():.4f} .. {np.sqrt(xi**2+eta).max():.4f}")


# --- drift ----
elif MODE == "drift":
    integ = os.environ.get("BH_INTEG", "rk45")
    h     = float(os.environ.get("BH_H", 1e-3))

    yaws   = np.linspace(0.05, 0.58, bh.width).astype(np.float32)
    bh.probe_yaw.from_numpy(yaws)
    bh.probe_pitch.from_numpy(np.full(bh.width, 0.25, np.float32))
    bh.probe_directions(CAM_D, LOOK_D)

    t0 = time.perf_counter()
    if integ == "rk45":
        bh.render_rays_diag(bh.r_ISCO)
    else:
        bh.render_rays_rk4(bh.r_ISCO, h)
    ti.sync()
    wall = time.perf_counter() - t0

    resid = bh.resid_max.to_numpy()
    rhs   = bh.rhs_count.to_numpy()[0, 0, :bh.width]
    st    = bh.state.to_numpy()[0, 0, :bh.width]
    done  = st != 0                      # 0 means it hit the step cap: discard

    tag = f"{integ}_{bh.tol:.0e}" if integ == "rk45" else f"rk4_{h:.0e}"
    np.savez(f"data/drift_{tag}.npz",
             integ=integ, tol=bh.tol, h=h, wall=wall,
             resid=resid, rhs=rhs, done=done, yaws=yaws)
    print(f"[drift] {tag:16s} median resid={np.median(resid[done]):.3e}  "
          f"median RHS={np.median(rhs[done]):.0f}  "
          f"completed={done.sum()}/{bh.width}")


# --- cost ----
elif MODE == "cost":
    integ = os.environ.get("BH_INTEG", "rk45")
    h     = float(os.environ.get("BH_H", 1e-3))
    W     = bh.width

    def fire(yaws):
        bh.probe_yaw.from_numpy(np.asarray(yaws, np.float32))
        bh.probe_pitch.from_numpy(np.full(W, 0.25, np.float32))
        bh.probe_directions(CAM_D, LOOK_D)

    lo, hi = 0.05, 0.58
    for _ in range(34):
        m = 0.5*(lo + hi)
        fire(np.full(W, m)); bh.render_rays_rk4(bh.r_ISCO, 1e-3); ti.sync()
        lo, hi = (m, hi) if bh.state.to_numpy()[0,0,0] == 1 else (lo, m)
    yc = 0.5*(lo + hi)

    eps  = np.logspace(-1, -6, W//2)
    yaws = np.concatenate([yc*(1 - eps), yc*(1 + eps)])
    fire(yaws)

    if integ == "rk45":
        bh.render_rays_diag(bh.r_ISCO); tag = f"rk45_{bh.tol:.0e}"
    else:
        bh.render_rays_rk4(bh.r_ISCO, h); tag = f"rk4_{h:.0e}"
    ti.sync()

    resid = bh.resid_max.to_numpy(); rhs = bh.rhs_count.to_numpy()[0,0,:W]
    st    = bh.state.to_numpy()[0,0,:W]
    ok    = np.isfinite(resid) & (resid > 0)

    np.savez(f"data/cost_{tag}.npz", integ=integ, tol=bh.tol, h=h, yc=yc,
             resid=resid, rhs=rhs, done=(st != 0), ok=ok, yaws=yaws,
             worst=resid[ok].max(), total=rhs[ok].sum())
    print(f"[cost] {tag:12s} yc={yc:.6f} worst={resid[ok].max():.3e} "
          f"total={rhs[ok].sum():.0f} capped={int((st==0).sum())}")


# ---- timing ----
elif MODE == "timing":
    arch = os.environ.get("BH_ARCH", "cuda")
    reps = int(os.environ.get("BH_REPS", 5))

    for _ in range(2):                  
        bh.inits(bh.init_camera_pos, bh.camera_dir, bh.FOV)
        bh.render_rays(); ti.sync()

    ts = []
    for _ in range(reps):
        bh.inits(bh.init_camera_pos, bh.camera_dir, bh.FOV); ti.sync()
        t0 = time.perf_counter()
        bh.render_rays(); ti.sync()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)

    np.savez(f"data/timing_{arch}_{bh.width}x{bh.height}.npz",
             arch=arch, width=bh.width, height=bh.height, times=ts,
             median=np.median(ts), q25=np.percentile(ts, 25),
             q75=np.percentile(ts, 75), tol=bh.tol)
    print(f"[timing] {arch:5s} {bh.width}x{bh.height}  "
          f"median={np.median(ts):.3f}s  "
          f"rays/s={5*bh.width*bh.height/np.median(ts):.3e}")

else:
    raise SystemExit(f"unknown BH_MODE={MODE}")
