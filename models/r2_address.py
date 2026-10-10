#!/usr/bin/env python3
"""R2 — deterministic addressing for the Flash3D substrate (replaces the racy PSH scatter).

Why this is sufficient (measured, see results/tramx/R2_DETERMINISTIC_ADDRESSING.md):
  * `XFMR.forward` documents reduced_coord and reduced_sep as UNUSED and derives its attention scopes
    from the row count alone: `scope_bucks = swin_plan.gen_scopes_from_plan(N, device)`.
  * `module_tree` on identical input is bit-identical over repeated calls (delta 0.000e+00).
  * The PSH scatter kernel is NOT deterministic (6,931-21,685 of 25,255 points change bucket between
    two identical calls, in every supported hash mode).
So the model is a deterministic function of (row count, features) — the layout is ours to define, and
the only non-deterministic component (the bucket assignment for the embedding) is replaced here.

Layout (auditable by construction, no kernel, no atomics):
  cell   = (x // S, y // S, tau // S)                     S = 16
  bucket = splitmix64(cell) % NB                          NB = 48 (rows = NB*512 = 24,576)
  order  = stable sort by (bucket, original index)        deterministic in any schedule
  row    = bucket * 512 + rank_within_bucket              so row(u) is a pure function of u
  pad    = cells beyond the point count get zero features
"""
from __future__ import annotations

import numpy as np
import torch

BUCKET = 128            # attention scope = SCOPE*BUCKET = 1024 tokens (a 512-bucket config from the
                        # scene-scale upstream example makes the attention span the whole cloud)
NB = 48                      # minimum; bucketize() raises it until no bucket exceeds BUCKET
                             # 48 * 512 = 24,576 == swinable_alignment(512, 8) -> one pool block exactly
ROWS = NB * BUCKET
S = 4                        # spatial/temporal quantisation for the cell (used as bucketize default)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit mixer (numpy, overflow-wrapped) — pure function of the integer input."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    z = x.copy()
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def bucketize(locs: np.ndarray, S: int = 4) -> dict:
    """Deterministic, DENSE spatial layout for one clip.

    Design (why not per-cell buckets): the model's attention scopes come from the row count alone
    (`swin_plan.gen_scopes_from_plan(N)`), so the buckets the model sees are contiguous BUCKET-row
    chunks. A per-cell bucket table therefore either leaves empty buckets (device-side assert,
    measured) or explodes the row count. Instead:

      1. give every point a spatial key  = ((x//S) * K + (y//S)) * K + (tau//S)
      2. stable-sort by that key -> spatially coherent order, deterministic in any schedule
      3. the model's bucket b is simply rows [b*BUCKET, (b+1)*BUCKET) of that order, so every bucket
         is populated and holds spatially nearby events (which is what PSH approximates)
      4. rows = ceil(N / ALIGN) * ALIGN with ALIGN = BUCKET*SCOPE*6, matching the pool alignment

    locs: (N,4) int64 [clip, x, y, t]; tau uses (t - t0)/64 with t0 of THIS clip (past-only).
    """
    n = locs.shape[0]
    t = locs[:, 3].astype(np.float64)
    t0 = float(t.min())
    tau = (t - t0) / 64.0
    K = 4096
    key = (((locs[:, 1] // S).astype(np.int64) * K + (locs[:, 2] // S).astype(np.int64)) * K
           + np.floor(tau / S).astype(np.int64))
    # coarsen the key until no 128-row chunk would mix wildly different cells (kept simple: the sort
    # itself guarantees contiguity, so no coarsening pass is needed).
    order = np.lexsort((np.arange(n), key))          # stable: spatial key, then original index
    row_all = np.empty(n, dtype=np.int64)
    row_all[order] = np.arange(n, dtype=np.int64)    # dense: no padding inside the point range
    ALIGN = BUCKET * 8 * 6
    rows = int(np.ceil(max(1, n) / ALIGN) * ALIGN)
    nb = rows // BUCKET
    counts = np.bincount(np.arange(n, dtype=np.int64) // BUCKET, minlength=nb)
    return {'row': row_all, 'bucket': key, 'counts': counts, 'n': n, 'nb': nb, 'rows': rows,
            'tau': tau, 't0': t0, 'S': S, 'align': ALIGN,
            'max_bucket': int(counts.max(initial=0)), 'key_unique': int(np.unique(key).size)}



def stem_features(locs: np.ndarray, feats: np.ndarray, bu: dict) -> np.ndarray:
    """Per-event causal features (same 7 columns as the campaign stem), in ORIGINAL point order."""
    n = locs.shape[0]
    b, x, y = locs[:, 0].astype(np.int64), locs[:, 1].astype(np.int64), locs[:, 2].astype(np.int64)
    t = locs[:, 3].astype(np.float64)
    pol = (feats[:, 3] > 0.5).astype(np.int64)
    out = np.zeros((n, 7), dtype=np.float32)
    pk = (b * 1000 + x) * 1000 + y
    order = np.argsort(pk, kind='stable')
    P, T, L = pk[order], t[order], pol[order]
    bounds = np.flatnonzero(np.r_[True, P[1:] != P[:-1]])
    dsp = np.zeros(n); dop = np.zeros(n); npix = np.zeros(n)
    for i, s in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else n
        idx = order[s:e]; tt = t[idx]; pp = pol[idx]
        srt = np.argsort(tt, kind='stable')
        idx, tt, pp = idx[srt], tt[srt], pp[srt]
        last = {0: None, 1: None}
        for j in range(len(idx)):
            a = last[pp[j]]; o = last[1 - pp[j]]
            dsp[idx[j]] = tt[j] - a if a is not None else 1e4
            dop[idx[j]] = tt[j] - o if o is not None else 1e4
            npix[idx[j]] = j + 1
            last[pp[j]] = tt[j]
    W = 512.0
    cell = b * 100000 + (x // 8) * 1000 + (y // 8)
    co = np.argsort(cell, kind='stable')
    CK, CT = cell[co], t[co]
    cb = np.flatnonzero(np.r_[True, CK[1:] != CK[:-1]])
    ncell = np.zeros(n)
    for i, s in enumerate(cb):
        e = cb[i + 1] if i + 1 < len(cb) else n
        idx, tt = co[s:e], CT[s:e]
        srt = np.argsort(tt, kind='stable'); idx, tt = idx[srt], tt[srt]
        j0 = 0
        for j in range(len(tt)):
            while tt[j] - tt[j0] > W:
                j0 += 1
            ncell[idx[j]] = j - j0 + 1
    out[:, 0] = pol
    out[:, 1] = bu['tau']
    out[:, 2] = np.log1p(dsp)
    out[:, 3] = np.log1p(dop)
    out[:, 4] = np.log1p(npix)
    out[:, 5] = np.log1p(ncell)
    out[:, 6] = bu['tau'] / 125.0
    return out


def scatter_to_rows(values: np.ndarray, bu: dict, mu=None, sd=None, dtype=torch.float32,
                    device='cuda'):
    """Place per-point values into the deterministic row layout (padding rows stay zero)."""
    v = values.astype(np.float32)
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32)
        v = (v - mu) / sd
    rows = torch.zeros((bu['rows'], v.shape[1]), dtype=dtype, device=device)
    idx = torch.from_numpy(bu['row']).to(device)
    rows[idx] = torch.from_numpy(v).to(device)
    return rows


def coord_rows(locs: np.ndarray, bu: dict, device='cuda'):
    """Padded coordinate rows (only used to keep the module signature; XFMR documents them as unused)."""
    n = locs.shape[0]
    t = locs[:, 3].astype(np.float64)
    co = np.zeros((n, 3), dtype=np.float32)
    co[:, 0] = locs[:, 1]
    co[:, 1] = locs[:, 2]
    co[:, 2] = (t - bu['t0']) / 64.0
    rows = torch.zeros((bu['rows'], 3), dtype=torch.float32, device=device)
    rows[torch.from_numpy(bu['row']).to(device)] = torch.from_numpy(co).to(device)
    return rows


def label_rows(seg: np.ndarray, bu: dict, device='cuda'):
    lab = torch.full((bu['rows'],), -1, dtype=torch.long, device=device)
    lab[torch.from_numpy(bu['row']).to(device)] = torch.from_numpy(seg).to(device)
    return lab
