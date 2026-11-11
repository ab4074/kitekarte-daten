"""Gezeitenströmung für die Kitekarte aus dem BSH-Zirkulationsmodell (HBM, Deutsche Bucht, ca. 900 m).

Lädt die frei verfügbaren 6-Stunden-Dateien der letzten DAYS Tage (CC BY 4.0, Quelle: BSH),
schneidet die Oberflächenströmung für die deutsche Nordseeküste aus und berechnet je Gitterpunkt
die Gezeitenanteile (harmonische Analyse). Die App kann daraus die Strömung für jede Uhrzeit
selbst ausrechnen, auch offline.

Umgebungsvariablen: DAYS (Standard 29), INSPECT=1 (nur Aufbau der ersten Datei ausgeben), OUT (Ausgabeordner).
"""
import os, sys, io, json, math, time, datetime as dt, urllib.request, urllib.error, tempfile
import numpy as np
import netCDF4 as nc

DAYS = int(os.environ.get('DAYS', '29'))
OUT = os.environ.get('OUT', 'data')
BOX = dict(lon0=7.75, lon1=10.05, lat0=53.35, lat1=55.15)     # deutsche Nordseeküste bis Hamburg
BASE = 'https://gdi.bsh.de/data/OpenData/OperationalModel/c_file_fine_{y}/c_file_fine_{y}{m:02d}{d:02d}{h:02d}_000_006.nc'
CONST = [('M2', 28.9841042), ('S2', 30.0), ('N2', 28.4397295), ('K1', 15.0410686), ('O1', 13.9430356),
         ('M4', 57.9682084), ('MS4', 58.9841042), ('M6', 86.9523127)]          # Grad pro Stunde
EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

def log(*a):
    print(*a, flush=True)

def fetch(url, path):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path, 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404: return False
            log('  HTTP', e.code, 'Versuch', k + 1)
        except Exception as e:
            log('  Fehler', e, 'Versuch', k + 1)
        time.sleep(5 * (k + 1))
    return False

def find_uv(ds):
    def score(v, want):
        sn = (getattr(v, 'standard_name', '') or '').lower(); ln = (getattr(v, 'long_name', '') or '').lower(); n = v.name.lower()
        if want == 'u':
            return ('eastward' in sn and 'velocity' in sn) * 3 + (n in ('u', 'uo', 'ucur', 'uu', 'uvel', 'u_velocity')) * 2 + ('east' in ln and ('current' in ln or 'velocity' in ln))
        return ('northward' in sn and 'velocity' in sn) * 3 + (n in ('v', 'vo', 'vcur', 'vv', 'vvel', 'v_velocity')) * 2 + ('north' in ln and ('current' in ln or 'velocity' in ln))
    cands = [v for v in ds.variables.values() if v.ndim >= 3]
    u = max(cands, key=lambda v: score(v, 'u')); vv = max(cands, key=lambda v: score(v, 'v'))
    if score(u, 'u') == 0 or score(vv, 'v') == 0: raise SystemExit('u/v nicht gefunden – bitte INSPECT=1 laufen lassen')
    return u, vv

def coords(ds, var):
    lat = lon = depth = tim = None
    for d in var.dimensions:
        dl = d.lower(); cv = ds.variables.get(d)
        sn = (getattr(cv, 'standard_name', '') or '').lower() if cv is not None else ''
        if 'lat' in dl or sn == 'latitude' or dl in ('y', 'ny'): lat = d
        elif 'lon' in dl or sn == 'longitude' or dl in ('x', 'nx'): lon = d
        elif 'time' in dl or sn == 'time': tim = d
        else: depth = d
    return tim, depth, lat, lon

def axis(ds, name, var, kind):
    if name in ds.variables: return np.asarray(ds.variables[name][:], dtype=float)
    # Koordinaten als eigene Variable (z. B. lat_u)
    for k, v in ds.variables.items():
        if v.dimensions == (name,) and kind in k.lower(): return np.asarray(v[:], dtype=float)
    raise SystemExit(f'Achse {name} nicht gefunden')

def inspect(path):
    ds = nc.Dataset(path)
    log('=== Aufbau', path)
    log('Dimensionen:', {k: len(v) for k, v in ds.dimensions.items()})
    for k, v in ds.variables.items():
        log(f'  {k} {v.dimensions} {v.shape} {v.dtype}', {a: str(getattr(v, a))[:80] for a in v.ncattrs()})
    log('Global:', {a: str(getattr(ds, a))[:120] for a in ds.ncattrs()})
    ds.close()

def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    now = dt.datetime.now(dt.timezone.utc)
    runs = []
    t = (now - dt.timedelta(days=DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    while t <= now:
        runs.append(t); t += dt.timedelta(hours=6)
    log('Läufe zu prüfen:', len(runs))
    series_t, series_u, series_v = [], [], []
    grid = None; ok = 0
    for r in runs:
        url = BASE.format(y=r.year, m=r.month, d=r.day, h=r.hour)
        p = os.path.join(tmp, 'f.nc')
        if not fetch(url, p): continue
        if os.environ.get('INSPECT') == '1':
            inspect(p); return
        ds = nc.Dataset(p)
        u, v = find_uv(ds)
        tn, dn, latn, lonn = coords(ds, u)
        tv = ds.variables[tn]
        times = nc.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'), only_use_cftime_datetimes=False)
        ts = np.array([(dt.datetime(x.year, x.month, x.day, x.hour, x.minute, x.second, tzinfo=dt.timezone.utc) - EPOCH).total_seconds() / 3600 for x in times])
        if grid is None:
            la = axis(ds, latn, u, 'lat'); lo = axis(ds, lonn, u, 'lon')
            j = np.where((la >= BOX['lat0']) & (la <= BOX['lat1']))[0]; i = np.where((lo >= BOX['lon0']) & (lo <= BOX['lon1']))[0]
            tv2, dv2, lav, lov = coords(ds, v)
            la_v = axis(ds, lav, v, 'lat'); lo_v = axis(ds, lov, v, 'lon')
            same = la_v.shape == la.shape and lo_v.shape == lo.shape and np.allclose(la_v, la) and np.allclose(lo_v, lo)
            grid = dict(j0=int(j[0]), j1=int(j[-1]) + 1, i0=int(i[0]), i1=int(i[-1]) + 1, lat=la[j[0]:j[-1] + 1], lon=lo[i[0]:i[-1] + 1], la_v=la_v, lo_v=lo_v, same=same)
            log('Gitter:', len(grid['lat']), 'x', len(grid['lon']), 'Breite', grid['lat'][0], grid['lat'][-1], 'Länge', grid['lon'][0], grid['lon'][-1],
                '| v auf eigenem Gitter:', not same, '| Tiefe-Dim:', dn, u.dimensions)
        sl = [slice(None)] * u.ndim
        di = u.dimensions.index
        if dn: sl[di(dn)] = 0
        sl[di(latn)] = slice(grid['j0'], grid['j1']); sl[di(lonn)] = slice(grid['i0'], grid['i1'])
        U = np.ma.filled(np.ma.asarray(u[tuple(sl)]).astype(np.float32), np.nan)
        # v aufs u-Gitter bringen (Arakawa-C: v liegt versetzt)
        tn2, dn2, lav, lov = coords(ds, v); sl2 = [slice(None)] * v.ndim
        if dn2: sl2[v.dimensions.index(dn2)] = 0
        if grid['same']:
            sl2[v.dimensions.index(lav)] = slice(grid['j0'], grid['j1']); sl2[v.dimensions.index(lov)] = slice(grid['i0'], grid['i1'])
            V = np.ma.filled(np.ma.asarray(v[tuple(sl2)]).astype(np.float32), np.nan)
        else:
            V = regrid(np.ma.filled(np.ma.asarray(v[tuple(sl2)]).astype(np.float32), np.nan), grid['la_v'], grid['lo_v'], grid['lat'], grid['lon'])
        ds.close(); os.remove(p)
        keep = ts < (ts[0] + 6 - 1e-6)
        series_t.append(ts[keep]); series_u.append(U[keep].astype(np.float16)); series_v.append(V[keep].astype(np.float16))
        ok += 1
        log(f'  {r:%Y-%m-%d %H} UTC: {keep.sum()} Zeitschritte')
    if not ok: raise SystemExit('keine Dateien gefunden')
    T = np.concatenate(series_t); order = np.argsort(T); T = T[order]
    U = np.concatenate(series_u)[order].astype(np.float32); V = np.concatenate(series_v)[order].astype(np.float32)
    _, uniq = np.unique(T, return_index=True); T, U, V = T[uniq], U[uniq], V[uniq]
    nt, ny, nx = U.shape
    log('Zeitreihe:', nt, 'Schritte über', round((T[-1] - T[0]) / 24, 1), 'Tage; Gitter', ny, 'x', nx)
    valid = np.isfinite(U) & np.isfinite(V)
    frac = valid.mean(0)
    water = frac >= 0.3                       # trockenfallendes Watt: trocken = keine Strömung
    U = np.where(valid, U, 0.0); V = np.where(valid, V, 0.0)
    span = (T[-1] - T[0]) / 24
    cons = []   # Rayleigh-Kriterium: zwei Anteile nur, wenn die Zeitreihe lang genug ist, um sie zu trennen
    for n, w in CONST:
        if all(abs(w - w2) * span * 24 / 360 >= 1.0 for _, w2 in cons): cons.append((n, w))
    log('Gezeitenanteile:', [n for n, _ in cons])
    A = [np.ones_like(T)]
    for n, w in cons:
        ph = np.deg2rad(w) * T; A += [np.cos(ph), np.sin(ph)]
    A = np.stack(A, 1)
    P = np.linalg.pinv(A)
    cells = np.where(water.ravel())[0]
    Yu = U.reshape(nt, -1)[:, cells]; Yv = V.reshape(nt, -1)[:, cells]
    Cu = P @ Yu; Cv = P @ Yv
    ru = Yu - A @ Cu; rv = Yv - A @ Cv
    expl = 1 - (ru.var(0) + rv.var(0)) / np.maximum(Yu.var(0) + Yv.var(0), 1e-9)
    log('Erklärte Varianz (Median, 10 %):', round(float(np.median(expl)), 3), round(float(np.percentile(expl, 10)), 3))
    m2u = np.hypot(Cu[1], Cu[2]); m2v = np.hypot(Cv[1], Cv[2])
    log('M2-Amplitude m/s (Median, Max):', round(float(np.median(np.hypot(m2u, m2v))), 3), round(float(np.max(np.hypot(m2u, m2v))), 3))
    # Ausgabe: Koeffizienten als int16 in mm/s, Reihenfolge je Zelle [u0, u(cos,sin)×K, v0, v(cos,sin)×K]
    K = 1 + 2 * len(cons)
    D = np.empty((len(cells), 2 * K), dtype=np.int16)
    D[:, :K] = np.clip(np.round(Cu.T * 1000), -32767, 32767); D[:, K:] = np.clip(np.round(Cv.T * 1000), -32767, 32767)
    lat = grid['lat']; lon = grid['lon']
    meta = dict(source='BSH, operationelles Zirkulationsmodell HBM (Deutsche Bucht, ca. 900 m), CC BY 4.0', made=now.isoformat(timespec='minutes'),
                period=[(EPOCH + dt.timedelta(hours=float(T[0]))).isoformat(timespec='minutes'), (EPOCH + dt.timedelta(hours=float(T[-1]))).isoformat(timespec='minutes')],
                epoch=EPOCH.isoformat(), unit='mm/s', consts=[[n, w] for n, w in cons], nx=nx, ny=ny,
                lat=[float(lat[0]), float(lat[-1])], lon=[float(lon[0]), float(lon[-1])], cells=len(cells),
                explained_median=round(float(np.median(expl)), 3))
    np.save(os.path.join(OUT, 'cells.npy'), cells.astype(np.int32))
    with open(os.path.join(OUT, 'stroemung.bin'), 'wb') as f:
        mask = np.zeros(ny * nx, dtype=np.uint8); mask[cells] = 1
        f.write(np.packbits(mask).tobytes()); f.write(D.tobytes())
    np.save(os.path.join(OUT, 'expl.npy'), expl.astype(np.float16))
    json.dump(meta, open(os.path.join(OUT, 'stroemung.json'), 'w'), ensure_ascii=False, indent=1)
    log('fertig:', meta)

def regrid(V, la_v, lo_v, lat, lon):
    """v vom versetzten Gitter linear auf das u-Gitter interpolieren (pro Zeitschritt, NaN-tolerant)."""
    Vn = np.nan_to_num(V, nan=0.0); M = np.isfinite(V).astype(np.float32)
    def interp_axis(A, src, dst, ax):
        if src[1] < src[0]: src = src[::-1]; A = np.flip(A, ax)
        idx = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2); w = (dst - src[idx]) / (src[idx + 1] - src[idx])
        a = np.take(A, idx, axis=ax); b = np.take(A, idx + 1, axis=ax)
        shp = [1] * A.ndim; shp[ax] = len(dst); w = w.reshape(shp)
        return a * (1 - w) + b * w
    full_lat_v = la_v; full_lon_v = lo_v
    if Vn.shape[1] != len(full_lat_v) or Vn.shape[2] != len(full_lon_v):
        raise SystemExit('v-Gitter passt nicht: ' + str(Vn.shape))
    Vi = interp_axis(interp_axis(Vn, full_lat_v, lat, 1), full_lon_v, lon, 2)
    Mi = interp_axis(interp_axis(M, full_lat_v, lat, 1), full_lon_v, lon, 2)
    return np.where(Mi > 0.5, Vi / np.maximum(Mi, 1e-6), np.nan).astype(np.float32)

if __name__ == '__main__':
    main()
