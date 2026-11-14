#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""gen_brdf_tables.py - writes engine/core/brdf_tables.h: how much light the GGX microfacet
lobe of core/brdf.h keeps after one bounce, for the multiple-scattering compensation
(Kulla & Conty 2017) and the diffuse layer under a dielectric's specular one.

    uv run tools/gen_brdf_tables.py

E(mu, r): the directional albedo of a white (F = 1) GGX lobe of roughness r (alpha = r^2),
height-correlated Smith masking, seen at cosine mu; E_avg(r) = 2 * integral of E(mu) mu dmu.
B(mu, r): the same with Schlick's Fresnel term (1 - v.h)^5 inside, so that a lobe of Fresnel
F0 + (1 - F0) (1 - v.h)^5 keeps F0 (E - B) + B (the split sum's scale and bias, Karis 2013).
32 x 32 nodes, mu and r from 0 to 1. Integrated by the visible normals: for microfacet
normals h drawn by their density G1(v) (v.h) D(h) / (n.v), the reflection's weight is
G2(v, l) / G1(v) (0 below the horizon); 128 x 128 stratified samples per node.
"""
import numpy as np

N = 32
S = 128


def g1(c, a):
    c = np.abs(c)
    return 2 * c / (c + np.sqrt(a * a + (1 - a * a) * c * c))


def vis(cv, cl, a):
    a2 = a * a
    d = cl * np.sqrt(cv * cv * (1 - a2) + a2) + cv * np.sqrt(cl * cl * (1 - a2) + a2)
    return 0.5 / d


def albedo(mu, r):
    a = max(r * r, 1e-4)
    mu = max(mu, 1e-4)
    v = np.array([np.sqrt(1 - mu * mu), 0.0, mu])
    u1, u2 = np.meshgrid((np.arange(S) + 0.5) / S, (np.arange(S) + 0.5) / S, indexing="ij")
    u1 = u1.ravel(); u2 = u2.ravel()
    vh = np.array([a * v[0], a * v[1], v[2]]); vh /= np.linalg.norm(vh)
    phi = 2 * np.pi * u1
    z = (1 - u2) * (1 + vh[2]) - vh[2]
    st = np.sqrt(np.maximum(0, 1 - z * z))
    hs = np.stack([st * np.cos(phi) + vh[0], st * np.sin(phi) + vh[1], z + vh[2]], -1)
    h = np.stack([a * hs[:, 0], a * hs[:, 1], np.maximum(hs[:, 2], 0)], -1)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    vdh = h @ v
    l = 2 * vdh[:, None] * h - v
    cl = l[:, 2]
    w = np.where(cl > 0, vis(mu, np.maximum(cl, 1e-9), a) * 4 * mu * np.maximum(cl, 0) / g1(mu, a), 0)
    return float(w.mean()), float((w * (1 - np.clip(vdh, 0, 1)) ** 5).mean())


def main():
    E = np.zeros((N, N))      # [r][mu]
    B = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            E[j, i], B[j, i] = albedo(i / (N - 1), j / (N - 1))
    mu = np.arange(N) / (N - 1)
    # the means: 2 int E mu dmu, by the trapezoid rule on a fine resampling of the nodes
    fine = np.linspace(0, 1, 2049)
    Eavg = np.array([2 * np.trapezoid(np.interp(fine, mu, E[j]) * fine, fine) for j in range(N)])
    Bavg = np.array([2 * np.trapezoid(np.interp(fine, mu, B[j]) * fine, fine) for j in range(N)])
    out = ["// brdf_tables.h - made by tools/gen_brdf_tables.py (do not edit): the share of the light a",
           "// white GGX lobe keeps after one bounce, for the multiple-scattering compensation of",
           "// core/brdf.h. BRDF_E[r][mu]: roughness r and cosine mu, both 0..1 in %d steps;" % N,
           "// BRDF_E_AVG[r]: its cosine-weighted mean over the hemisphere. BRDF_B, BRDF_B_AVG: the",
           "// same with Schlick's (1 - v.h)^5 inside: a lobe of Fresnel F0 keeps F0 (E - B) + B.",
           "#ifndef LR_BRDF_TABLES_H", "#define LR_BRDF_TABLES_H", "",
           "#define BRDF_TABLE_N %d" % N, "",
           "static const float BRDF_E[%d][%d] = {" % (N, N)]
    for j in range(N):
        out.append("    {" + ", ".join("%.6ff" % x for x in E[j]) + "},")
    out.append("};")
    out.append("")
    out.append("static const float BRDF_E_AVG[%d] = {" % N)
    out.append("    " + ", ".join("%.6ff" % x for x in Eavg))
    out.append("};")
    out.append("")
    out.append("static const float BRDF_B[%d][%d] = {" % (N, N))
    for j in range(N):
        out.append("    {" + ", ".join("%.6ff" % x for x in B[j]) + "},")
    out.append("};")
    out.append("")
    out.append("static const float BRDF_B_AVG[%d] = {" % N)
    out.append("    " + ", ".join("%.6ff" % x for x in Bavg))
    out.append("};")
    out.append("")
    out.append("#endif")
    with open("engine/core/brdf_tables.h", "w") as f:
        f.write("\n".join(out) + "\n")
    print("E(1, r):", np.round(E[:, -1][::4], 3))
    print("E_avg:", np.round(Eavg[::4], 3))


if __name__ == "__main__":
    main()
