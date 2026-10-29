"""
This code reads the data and writes the five figures into figures/.

    python make_figures.py
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 160,"axes.grid": True, "grid.alpha": 0.25})

SUMMARY = []
def note(s):
    print(s); SUMMARY.append(s)

def inventory():
    g = {k: sorted(glob.glob(f"data/{k}*.npz"))
         for k in ("shadow_", "drift_", "timing_")}
    print("data/ inventory:")
    for k, v in g.items():
        print(f"  {k:9s} {len(v):3d} file(s)")
    if not g["shadow_"]:
        print("  !! no shadow data  -> run:  python run_all.py shadow")
    if not g["drift_"]:
        print("  !! no drift data   -> run:  python run_all.py drift"
              "     (this is what makes figures 2 and 3)")
    if not g["timing_"]:
        print("  !! no timing data  -> run:  python run_all.py timing")
    print()
    return g


def critical_xi_eta(a, n=200000):
    """Bardeen critical curve, parametrised by spherical photon orbit radius."""
    if a < 1e-3:                                    # a -> 0 forms are 0/0
        t = np.linspace(0, np.pi, n)
        return np.zeros(n), np.full(n, 27.0), np.full(n, 3.0)
    r1 = 2*(1 + np.cos((2/3)*np.arccos(-a)))        # prograde photon orbit
    r2 = 2*(1 + np.cos((2/3)*np.arccos( a)))        # retrograde
    r  = np.linspace(r1, r2, n)
    xi  = -(r**3 - 3*r**2 + a**2*r + a**2) / (a*(r - 1))
    eta = -r**3*(r**3 - 6*r**2 + 9*r - 4*a**2) / (a**2*(r - 1)**2)
    ok  = eta >= 0
    return xi[ok], eta[ok], r[ok]


def closed_curve(a):
    """Analytic critical curve as a closed (alpha, beta) loop, theta_o = pi/2."""
    xi, eta, _ = critical_xi_eta(a)
    alpha = np.concatenate([-xi,  -xi[::-1]])
    beta  = np.concatenate([np.sqrt(eta), -np.sqrt(eta)[::-1]])
    return alpha, beta


def metrics(alpha, beta):
    """Shape metrics for an ordered closed boundary."""
    x, y   = alpha, beta
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x*yn - xn*y
    A  = 0.5*cr.sum()
    cx = ((x + xn)*cr).sum()/(6*A)
    cy = ((y + yn)*cr).sum()/(6*A)
    A  = abs(A)
    Req = np.sqrt(A/np.pi)
    b   = np.hypot(x - cx, y - cy)
    return dict(Req=Req, area=A, cx=cx, cy=cy,
                displacement=abs(cx)/Req,
                flattening=(b.max() - b.min())/b.mean(),
                flattening_robust=(np.percentile(b, 98) -
                                   np.percentile(b, 2))/b.mean())


def load_shadows():
    out = []
    for f in sorted(glob.glob("data/shadow_a*.npz")):
        d = np.load(f)
        a, psis, xi, eta = float(d["a"]), d["psis"], d["xi"], d["eta"]

        beta = np.sign(np.sin(psis)) * np.sqrt(np.clip(eta, 0, None))
        br   = d["bracket"]

        good = br < 1e-4
        if good.sum() < 8:
            good = np.ones_like(br, dtype=bool)
        out.append(dict(a=a, psis=psis, xi=xi, eta=eta,
                        alpha=-xi, beta=beta, bracket=br, good=good))
    return sorted(out, key=lambda r: r["a"])


# FIGURE 1: Bardeen comparison ==
def figure_bardeen(shadows):
    if not shadows:
        print("FIG 1 SKIPPED: no shadow data.")
        return
    show = [s for s in shadows if s["a"] in
            {min(s2["a"] for s2 in shadows), 0.5, 0.9, 0.998}] or shadows[::4]

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))

    for i, s in enumerate(show):
        c = f"C{i}"
        al, be = closed_curve(s["a"])
        ax[0].plot(al, be, "-", lw=1.2, color=c, alpha=0.55,
                   label=f"$a$={s['a']:.3f}")
        g = s["good"]
        ax[0].plot(s["alpha"][g], s["beta"][g], ".", ms=2.5, color=c)
        if (~g).any():
            ax[0].plot(s["alpha"][~g], s["beta"][~g], "x", ms=4, color="k",
                       mew=0.6, alpha=0.7)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$\alpha\ /\ M$"); ax[0].set_ylabel(r"$\beta\ /\ M$")
    ax[0].set_title("Shadow boundary: measured (dots) vs Bardeen (lines)")
    ax[0].legend(fontsize=6.5, loc="upper left")

    ndrop = 0
    for s in shadows:
        g = s["good"]; ndrop += int((~g).sum())
        xi_a, eta_a, _ = critical_xi_eta(s["a"])
        o = np.argsort(xi_a)
        eta_fit = np.interp(s["xi"][g], xi_a[o], eta_a[o])
        b_meas  = np.sqrt(s["xi"][g]**2 + np.clip(s["eta"][g], 0, None))
        b_pred  = np.sqrt(s["xi"][g]**2 + np.clip(eta_fit, 0, None))
        err = np.abs(b_meas - b_pred)
        if err.size == 0:
            continue
        ax[1].semilogy(s["a"], np.median(err), "o", ms=4, color="C0")
        ax[1].semilogy([s["a"], s["a"]],
                       [np.percentile(err, 10), np.percentile(err, 90)],
                       "-", lw=1, color="C0", alpha=0.5)
    ax[1].set_xlabel("spin $a/M$")
    ax[1].set_ylabel(r"$|b_{\rm meas} - b_{\rm Bardeen}|\ /\ M$")
    ax[1].set_title("Deviation from the analytic critical curve")

    fig.tight_layout(); fig.savefig("figures/fig1_bardeen.png"); plt.close(fig)
    print("  wrote figures/fig1_bardeen.png")

    note(f"FIG 1  {ndrop} of {sum(len(s['psis']) for s in shadows)} probe angles "
         f"failed to converge and were excluded (marked x)")
    s0 = shadows[0]; g0 = s0["good"]
    b0 = np.sqrt(s0["xi"][g0]**2 + np.clip(s0["eta"][g0], 0, None))
    note(f"FIG 1  a={s0['a']:.3f}: measured b_crit = {b0.mean():.6f} "
         f"+/- {b0.std():.2e} M   |   analytic 3*sqrt(3) = {3*np.sqrt(3):.6f} M   "
         f"|   error = {100*abs(b0.mean()-3*np.sqrt(3))/(3*np.sqrt(3)):.4f}%")


# FIGURE 2: drift vs tolerance (RK45)
def load_drift():
    """A ray that hit the step cap still accumulated a valid maximum residual
    along the path it did traverse, so it is kept -- only rays whose residual
    is non-finite or non-positive are unusable on a log axis."""
    rk45, rk4 = [], []
    for f in sorted(glob.glob("data/drift_*.npz")):
        d = np.load(f)
        resid, rhs, done = d["resid"], d["rhs"], d["done"]
        usable = np.isfinite(resid) & (resid > 0)
        if usable.sum() < 4:
            print(f"  skipped {os.path.basename(f)}: only {usable.sum()} rays "
                  f"with a positive finite residual (all zero? kernel not run?)")
            continue
        if done.sum() == 0:
            print(f"  note   {os.path.basename(f)}: no ray terminated before the "
                  f"step cap -- residuals still used, but raise BH_STEPS")
        rec = dict(resid=np.median(resid[usable]),
                   rhs=np.median(rhs[usable]),
                   p25=np.percentile(resid[usable], 25),
                   p75=np.percentile(resid[usable], 75),
                   wall=float(d["wall"]),
                   n=int(usable.sum()), ndone=int(done.sum()))
        if str(d["integ"].item()) == "rk45":
            rec["x"] = float(d["tol"]); rk45.append(rec)
        else:
            rec["x"] = float(d["h"]);   rk4.append(rec)
    return (sorted(rk45, key=lambda r: r["x"]),
            sorted(rk4,  key=lambda r: r["x"]))


def figure_drift(rk45):
    if len(rk45) < 2:
        print(f"FIG 2 SKIPPED: need >=2 usable rk45 runs, have {len(rk45)}.")
        return
    tol = np.array([r["x"] for r in rk45])
    med = np.array([r["resid"] for r in rk45])
    lo  = np.array([r["p25"] for r in rk45])
    hi  = np.array([r["p75"] for r in rk45])

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.fill_between(tol, lo, hi, alpha=0.2, color="C0", label="interquartile range")
    ax.loglog(tol, med, "o-", color="C0", label="median over 128 rays")

    kmin = int(np.argmin(med))

    m = np.zeros_like(med, dtype=bool); m[kmin:] = True
    m &= med > 1.5*med.min()
    ax.axhline(med.min(), color="C3", ls=":", lw=0.9)
    ax.text(tol[-1], med.min()*1.25, f"float32 floor  {med.min():.1e}",
            color="C3", fontsize=6.5, va="bottom", ha="left")
    if m.sum() >= 3:
        p = np.polyfit(np.log10(tol[m]), np.log10(med[m]), 1)
        ax.loglog(tol[m], 10**np.polyval(p, np.log10(tol[m])), "k--", lw=0.9,
                  label=f"slope = {p[0]:.2f}")
        note(f"FIG 2  drift scales as tol^{p[0]:.2f} "
             f"(ideal 1.00; a flat tail is the float32 roundoff floor)")

    ax.set_xlabel("integrator tolerance"); ax.invert_xaxis()
    ax.set_ylabel(r"max relative drift in $p_r^2 - R(r)$")
    ax.set_title("First-integral drift vs tolerance")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/fig2_drift_vs_tol.png"); plt.close(fig)
    print("  wrote figures/fig2_drift_vs_tol.png")

    note(f"FIG 2  best drift = {med.min():.3e} at tol = {tol[np.argmin(med)]:.0e}")


# FIGURE 3: RK4 vs RK45 (cost) 
def load_cost():
    rk45, rk4 = [], []
    for f in sorted(glob.glob("data/cost_*.npz")):
        d = np.load(f)
        rec = dict(worst=float(d["worst"]), total=float(d["total"]))
        if str(d["integ"].item()) == "rk45":
            rec["x"] = float(d["tol"]); rk45.append(rec)
        else:
            rec["x"] = float(d["h"]);   rk4.append(rec)
    return (sorted(rk45, key=lambda r: -r["worst"]),
            sorted(rk4,  key=lambda r: -r["worst"]))


def figure_rk4_vs_rk45(rk45, rk4):

    c45, c4 = load_cost()
    if len(c45) < 2 or len(c4) < 2:
        print(f"FIG 3 SKIPPED: need >=2 cost runs of each, have "
              f"{len(c45)} rk45 and {len(c4)} rk4.  Run: python run_all.py cost")
        return

    def pareto(rec):

        keep, best = [], np.inf
        for r in sorted(rec, key=lambda r: r["total"]):
            if r["worst"] < best:
                keep.append(r); best = r["worst"]
        return keep

    p4, p45 = pareto(c4), pareto(c45)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for rec, front, lab, c, mk in (
            (c4,  p4,  "fixed-step RK4", "C3", "s"),
            (c45, p45, "adaptive Dormand-Prince RK45", "C0", "o")):
        dom = [r for r in rec if r not in front]
        if dom:
            ax.loglog([r["total"] for r in dom], [r["worst"] for r in dom],
                      mk, color=c, ms=4.5, mfc="none", alpha=0.45)
        ax.loglog([r["total"] for r in front], [r["worst"] for r in front],
                  mk + "-", color=c, ms=4.5, lw=1.4, label=lab)
    ax.plot([], [], "ko", ms=4.5, mfc="none", alpha=0.5,
            label="past the roundoff floor")

    ax.set_xlabel("total derivative evaluations for the ray fan")
    ax.set_ylabel(r"worst-case relative drift in $p_r^2 - R(r)$")
    ax.set_title("Cost to render the photon ring to a given accuracy")
    ax.legend(fontsize=7, loc="upper right")

    def cost_at(rec, target):
        x = np.log10([r["total"] for r in rec]); y = np.log10([r["worst"] for r in rec])
        o = np.argsort(-y)
        if not (y.min() <= np.log10(target) <= y.max()):
            return None
        return 10**np.interp(-np.log10(target), -y[o], x[o])

    for r in p4:
        c5 = cost_at(p45, r["worst"])
        if c5:
            ax.annotate("", xy=(c5, r["worst"]), xytext=(r["total"], r["worst"]),
                        arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.7))
            note(f"FIG 3  worst-case {r['worst']:.2e}: RK4 {r['total']:.3g} evals, "
                 f"RK45 {c5:.3g}  ->  {r['total']/c5:.1f}x")
    best4 = min(r["worst"] for r in c4); best5 = min(r["worst"] for r in c45)
    b4 = min(p4, key=lambda r: r["worst"])
    c5b = cost_at(p45, b4["worst"])
    if c5b:
        note(f"FIG 3  HEADLINE -- at RK4's best accuracy ({b4['worst']:.2e}) "
             f"RK4 needs {b4['total']:.3g} evals, RK45 {c5b:.3g}  ->  "
             f"{b4['total']/c5b:.1f}x")
    if best5 < best4:
        note(f"FIG 3  RK45 reaches {best5:.2e}, an accuracy RK4 never attains "
             f"(best {best4:.2e}) at any step size tested")

    fig.tight_layout(); fig.savefig("figures/fig3_rk4_vs_rk45.png"); plt.close(fig)
    print("  wrote figures/fig3_rk4_vs_rk45.png")


# FIGURE 4: CPU vs GPU
def figure_timing():
    runs = {}
    for f in sorted(glob.glob("data/timing_*.npz")):
        d = np.load(f)
        runs.setdefault(str(d["arch"]), []).append(
            (int(d["width"])*int(d["height"]), float(d["median"]),
             float(d["q25"]), float(d["q75"]), int(d["width"]), int(d["height"])))
    if not runs:
        print("FIG 4 SKIPPED: no timing data.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for arch, c in (("cpu", "C3"), ("cuda", "C0")):
        if arch not in runs:
            continue
        v = np.array(sorted(runs[arch]))
        px, med, q25, q75 = v[:,0], v[:,1], v[:,2], v[:,3]
        ax[0].errorbar(px, med, yerr=[med-q25, q75-med], fmt="o-", ms=4,
                       color=c, capsize=2,
                       label={"cpu":"CPU (ti.cpu)","cuda":"GPU (ti.cuda)"}[arch])
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("pixels per frame"); ax[0].set_ylabel("wall-clock per frame / s")
    ax[0].set_title("Render time, identical kernels")
    ax[0].legend(fontsize=7)

    if "cpu" in runs and "cuda" in runs:
        cpu = {int(r[0]): r[1] for r in runs["cpu"]}
        gpu = {int(r[0]): r[1] for r in runs["cuda"]}
        shared = sorted(set(cpu) & set(gpu))
        if shared:
            sp = [cpu[p]/gpu[p] for p in shared]
            ax[1].bar(range(len(shared)), sp, color="C2", width=0.55)
            ax[1].set_xticks(range(len(shared)))
            ax[1].set_xticklabels([f"{p/1000:.0f}k" for p in shared])
            ax[1].set_xlabel("pixels per frame"); ax[1].set_ylabel("speedup  CPU / GPU")
            ax[1].set_title("GPU speedup factor")
            for i, s in enumerate(sp):
                ax[1].text(i, s, f"{s:.0f}x", ha="center", va="bottom", fontsize=8)
            note(f"FIG 4  GPU speedup {min(sp):.0f}x - {max(sp):.0f}x "
                 f"across {len(shared)} resolutions")

    fig.tight_layout(); fig.savefig("figures/fig4_cpu_vs_gpu.png"); plt.close(fig)
    print("  wrote figures/fig4_cpu_vs_gpu.png")


# FIGURE 5: asymmetry vs spin
def on_psi_grid(alpha, beta, psis):
    """Resample a closed boundary onto the probe's own angular grid. The
    percentile metrics depend on how densely the curve is sampled, so the
    analytic curve must be sampled exactly the way the bisection samples it."""
    ang = np.arctan2(beta, alpha) % (2*np.pi)
    rad = np.hypot(alpha, beta)
    o = np.argsort(ang)
    R = np.interp(psis, ang[o], rad[o], period=2*np.pi)
    return R*np.cos(psis), R*np.sin(psis)


def figure_asymmetry(shadows):
    if len(shadows) < 3:
        print(f"FIG 5 SKIPPED: need >=3 spins, have {len(shadows)}.")
        return
    a_m  = np.array([s["a"] for s in shadows])
    meas = [metrics(s["alpha"][s["good"]], s["beta"][s["good"]]) for s in shadows]
    psi_ref = shadows[0]["psis"]
    a_th = np.linspace(0.001, 0.998, 120)
    theo = [metrics(*on_psi_grid(*closed_curve(a), psi_ref)) for a in a_th]

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5))
    for i, (key, lab) in enumerate((
            ("displacement", "centroid displacement  $|\\alpha_c| / R_{\\rm eq}$"),
            ("flattening_robust", "flattening  $(b_{98\\%}-b_{2\\%})/\\bar b$"),
            ("area",         "shadow area  $/\\ M^2$"))):
        ax[i].plot(a_th, [t[key] for t in theo], "-", color="C7", lw=1,
                   label="analytic")
        ax[i].plot(a_m, [m[key] for m in meas], "o", ms=4, color="C0",
                   label="measured")
        ax[i].set_xlabel("spin  $a/M$"); ax[i].set_title(lab, fontsize=8.5)
        ax[i].legend(fontsize=7)
    ax[2].set_ylim(0, None)

    fig.suptitle(r"Shadow asymmetry scaling with spin ($\theta_{\rm obs}=90^\circ$)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig("figures/fig5_asymmetry.png"); plt.close(fig)
    print("  wrote figures/fig5_asymmetry.png")

    d = np.array([m["displacement"] for m in meas])
    ar = np.array([m["area"] for m in meas])
    note(f"FIG 5  displacement {d.min():.3f} -> {d.max():.3f} over the spin range")
    note(f"FIG 5  area varies only {100*(ar.max()-ar.min())/ar.max():.1f}% "
         f"({ar.max():.1f} -> {ar.min():.1f} M^2) -- shape distorts, area barely moves")


if __name__ == "__main__":
    inventory()
    sh = load_shadows()
    rk45, rk4 = load_drift()
    figure_bardeen(sh)
    figure_drift(rk45)
    figure_rk4_vs_rk45(rk45, rk4)
    figure_timing()
    figure_asymmetry(sh)

    print("\n" + "="*70 + "\nNUMBERS FOR THE REPORT\n" + "="*70)
    for s in SUMMARY:
        print("  " + s)
    with open("figures/summary.txt", "w") as f:
        f.write("\n".join(SUMMARY) + "\n")
