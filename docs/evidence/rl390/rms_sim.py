# rl#390: composed descent RMS vs footprint, old floor/clamp vs new.
import numpy as np
exec(open('noise_sim.py').read().split('rng = np.random')[0])  # reuse hash/noise defs

def smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1); return t*t*(3-2*t)
def grain_fade(wl, fw): return 1 - smoothstep(0.07*wl, 0.28*wl, fw)

def color_rms(fw, floor):
    tot, wl, amp = 0.0, 2.6, 1.0
    while wl > floor:
        fade = grain_fade(wl, fw)
        if fade <= 0.001: break
        tot += (amp*fade)**2 * 0.294  # gnoise field variance (2.1x-scaled RMS 0.542^2)
        wl /= 3; amp *= 0.82
    return np.sqrt(tot)

def relief_rms(fw, floor):
    tot, nwl, nw = 0.0, 0.45, 0.0405
    grad_unit = 2.297  # gnoise fd-gradient RMS at 0.27wl step, per unit wl (from noise_sim)
    while nwl > floor:
        fade = grain_fade(nwl, fw)
        if fade <= 0.001: break
        tot += (fade * nw * grad_unit / nwl)**2
        nwl /= 3; nw *= 0.45
    return np.sqrt(tot)

print(f"{'fw':>8s} {'colorOLD':>9s} {'colorNEW':>9s} {'ratio':>6s}   {'reliefOLD':>9s} {'reliefNEW':>9s} {'ratio':>6s}")
for fw_mm in [0.02, 0.05, 0.1, 0.2, 0.5, 1, 3, 10]:
    fw = fw_mm/1000
    co = color_rms(max(fw,1e-4), 0.0035); cn = color_rms(max(fw,1e-5), 1e-4)
    ro = relief_rms(max(fw,1e-4), 0.0035); rn = relief_rms(max(fw,1e-5), 1e-4)
    print(f"{fw_mm:6.2f}mm {co:9.4f} {cn:9.4f} {cn/co:6.2f}   {ro:9.4f} {rn:9.4f} {rn/ro:6.2f}")
