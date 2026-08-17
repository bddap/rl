import numpy as np

M = np.uint64(0xffffffff)
def h2(x, y, seed):
    x = np.asarray(x).astype(np.int64).astype(np.uint64) & M; y = np.asarray(y).astype(np.int64).astype(np.uint64) & M; s = np.uint64(seed)
    h = (x * np.uint64(0x8da6b343)) & M
    h ^= (y * np.uint64(0xd8163841)) & M
    h ^= (s * np.uint64(0xcb1ab31f)) & M
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0x165667b1)) & M
    h ^= h >> np.uint64(16)
    return h

def rand01(h): return (h & np.uint64(0xffffff)).astype(np.float64) / 0x1000000

def quint(f): return f*f*f*(f*(f*6-15)+10)

def vnoise(px, py, seed):
    ix = np.floor(px).astype(np.int64); iy = np.floor(py).astype(np.int64)
    fx = px - ix; fy = py - iy
    wx = quint(fx); wy = quint(fy)
    a = rand01(h2(ix, iy, seed)); b = rand01(h2(ix+1, iy, seed))
    c = rand01(h2(ix, iy+1, seed)); d = rand01(h2(ix+1, iy+1, seed))
    return 2*((a*(1-wx)+b*wx)*(1-wy) + (c*(1-wx)+d*wx)*wy) - 1

def grad2(h):
    a = (h & np.uint64(0xffff)).astype(np.float64) * (2*np.pi/65536)
    return np.cos(a), np.sin(a)

def gnoise(px, py, seed, scale=1.0):
    ix = np.floor(px).astype(np.int64); iy = np.floor(py).astype(np.int64)
    fx = px - ix; fy = py - iy
    wx = quint(fx); wy = quint(fy)
    def corner(dx, dy):
        gx, gy = grad2(h2(ix+dx, iy+dy, seed))
        return gx*(fx-dx) + gy*(fy-dy)
    a = corner(0,0); b = corner(1,0); c = corner(0,1); d = corner(1,1)
    return scale*((a*(1-wx)+b*wx)*(1-wy) + (c*(1-wx)+d*wx)*wy)

rng = np.random.default_rng(7)
N = 2_000_000
px = rng.uniform(0, 4096, N); py = rng.uniform(0, 4096, N)

v = vnoise(px, py, 51)
g = gnoise(px, py, 51)
print("field RMS  vnoise=%.4f gnoise(unscaled)=%.4f  scale-to-match=%.4f" % (
    np.sqrt((v*v).mean()), np.sqrt((g*g).mean()),
    np.sqrt((v*v).mean())/np.sqrt((g*g).mean())))

# relief finite-diff gradient, step = 0.27 (cell units, wl=1)
s = 0.27
def fd(f, seed, scale=None):
    if scale is None:
        h0 = f(px, py, seed); hx = f(px+s, py, seed); hz = f(px, py+s, seed)
    else:
        h0 = f(px, py, seed, scale); hx = f(px+s, py, seed, scale); hz = f(px, py+s, seed, scale)
    return np.sqrt(((hx-h0)**2 + (hz-h0)**2).mean()) / s

gv = fd(vnoise, 51)
sc = np.sqrt((v*v).mean())/np.sqrt((g*g).mean())
gg = fd(gnoise, 51, sc)
print("fd-grad RMS  vnoise=%.4f gnoise(field-matched)=%.4f  nweight ratio=%.4f" % (gv, gg, gv/gg))
