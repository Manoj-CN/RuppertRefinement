"""
CDT mesh visualizer
===================
Reads a triangulation in OFF format (optionally overlaid with constraint edges
from a CDT input .txt file) and displays it with matplotlib.

Usage
-----
# Just the triangulation (OFF file only):
    python visualize_mesh.py output.off

# Triangulation + constraint edges highlighted:
    python visualize_mesh.py output.off --constraints inputs/island.txt

# Save to image instead of showing interactively:
    python visualize_mesh.py output.off --save mesh.png

Producing the OFF file from C++
---------------------------------
After calling refineRuppert (or just insertVertices/insertEdges), add:

    #include <fstream>
    void saveMeshOff(const std::string& path,
                     const CDT::Triangulation<double>& cdt)
    {
        std::ofstream f(path);
        f.precision(17);
        f << "OFF\\n";
        f << cdt.vertices.size() << ' ' << cdt.triangles.size() << " 0\\n";
        for (const auto& v : cdt.vertices)
            f << v.x << ' ' << v.y << " 0\\n";
        for (const auto& t : cdt.triangles)
            f << "3 " << t.vertices[0] << ' '
              << t.vertices[1] << ' ' << t.vertices[2] << "\\n";
    }
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def read_off(path):
    """Return (vertices, triangles, fixed_edges) from a 2-D OFF file.

    vertices    – (N, 2) float array
    triangles   – (M, 3) int array  (raw CDT indices, incl. super-triangle)
    fixed_edges – (K, 2) int array  (CDT internal indices), or empty array
                  when the file has no FIXED_EDGES section.
    """
    with open(path) as f:
        all_lines = [l.rstrip() for l in f if l.strip()]

    # Separate standard OFF lines from comment lines
    off_lines = [l for l in all_lines if not l.startswith('#')]
    comment_lines = [l for l in all_lines if l.startswith('#')]

    if off_lines[0] != "OFF":
        raise ValueError(f"'{path}' is not an OFF file (first line: {off_lines[0]!r})")

    n_verts, n_faces, _ = map(int, off_lines[1].split())

    verts = []
    for i in range(n_verts):
        parts = off_lines[2 + i].split()
        verts.append((float(parts[0]), float(parts[1])))

    tris = []
    for i in range(n_faces):
        parts = off_lines[2 + n_verts + i].split()
        assert int(parts[0]) == 3, "Only triangular faces supported"
        tris.append((int(parts[1]), int(parts[2]), int(parts[3])))

    # Parse fixed edges written by SaveToOff.h:
    #   # FIXED_EDGES <count>
    #   # <v1> <v2>
    fixed_edges = []
    fe_count = None
    for line in comment_lines:
        parts = line[1:].split()  # strip leading '#'
        if not parts:
            continue
        if parts[0] == 'FIXED_EDGES':
            fe_count = int(parts[1])
        elif fe_count is not None:
            fixed_edges.append((int(parts[0]), int(parts[1])))

    return (np.array(verts, dtype=float),
            np.array(tris, dtype=int),
            np.array(fixed_edges, dtype=int) if fixed_edges else np.empty((0, 2), dtype=int))


def read_cdt_input(path):
    """Return (vertices, edges) from a CDT input .txt file.

    vertices – (N, 2) float array  (user vertices, 0-based)
    edges    – (M, 2) int array    (0-based user indices, same as in the file)
    """
    with open(path) as f:
        tokens = f.read().split()

    idx = 0
    n_verts = int(tokens[idx]); idx += 1
    n_edges = int(tokens[idx]); idx += 1

    verts = []
    for _ in range(n_verts):
        x = float(tokens[idx]); idx += 1
        y = float(tokens[idx]); idx += 1
        verts.append((x, y))

    edges = []
    for _ in range(n_edges):
        v1 = int(tokens[idx]); idx += 1
        v2 = int(tokens[idx]); idx += 1
        edges.append((v1, v2))

    return np.array(verts, dtype=float), np.array(edges, dtype=int)


# ---------------------------------------------------------------------------
# Super-triangle detection
# ---------------------------------------------------------------------------

def _super_triangle_present(verts):
    """Return True when the file still contains the CDT super-triangle.

    CDT places the super-triangle at vertex indices 0, 1, 2 with coordinates
    outside the user-vertex bounding box.  After eraseOuterTriangles() those
    three vertices are removed and indices are shifted down by 3, so 0-2
    become ordinary boundary vertices that lie inside the bounding box.

    Detection: if any of vertices 0-2 lies more than one full domain span
    outside the bounding box of the remaining vertices, it must be a
    super-triangle vertex.  CDT places super-triangle vertices 3-4x the
    domain span away from the centroid, so a 1x-span threshold cleanly
    separates them from ordinary boundary vertices that merely happen to
    be the extreme point on one axis.
    """
    if len(verts) < 4:
        return False
    rest = verts[3:]
    x_min, x_max = rest[:, 0].min(), rest[:, 0].max()
    y_min, y_max = rest[:, 1].min(), rest[:, 1].max()
    span = max(x_max - x_min, y_max - y_min, 1e-9)
    return any(
        verts[i, 0] < x_min - span or verts[i, 0] > x_max + span or
        verts[i, 1] < y_min - span or verts[i, 1] > y_max + span
        for i in range(3)
    )


def _domain_tris(verts, tris):
    """Return only domain triangles (drop super-triangle-touching ones if present)."""
    if _super_triangle_present(verts):
        mask = np.ones(len(tris), dtype=bool)
        for si in (0, 1, 2):
            mask &= ~np.any(tris == si, axis=1)
        return tris[mask]
    return tris  # super-triangle already erased; all triangles are domain


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(off_path, constraints_path=None, save_path=None):
    verts, tris, fixed_edges = read_off(off_path)

    domain_tris = _domain_tris(verts, tris)
    if len(domain_tris) == 0:
        print("Warning: no domain triangles found.")
        domain_tris = tris

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_title(f"CDT mesh  —  {len(domain_tris)} triangles, "
                 f"{len(verts) - 3} user vertices")

    # --- triangles (filled lightly, edges drawn) ---
    # Draw each triangle explicitly rather than using triplot/tripcolor so
    # that no edges are skipped due to matplotlib's internal re-triangulation
    # of the super-triangle vertex coordinates.
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection, LineCollection
    patches = []
    for t in domain_tris:
        patches.append(Polygon(verts[t], closed=True))
    ax.add_collection(PatchCollection(patches, facecolor='#ddeeff',
                                      edgecolor='none', alpha=1.0, zorder=1))
    # Draw every edge explicitly
    segments = []
    for t in domain_tris:
        for i in range(3):
            a, b = verts[t[i]], verts[t[(i + 1) % 3]]
            segments.append([a, b])
    ax.add_collection(LineCollection(segments, colors='steelblue',
                                     linewidths=0.5, alpha=0.8, zorder=2))

    # --- constraint edges ---
    # Prefer the fixed edges embedded in the OFF file (written by SaveToOff.h
    # from cdt.fixedEdges).  These reflect the actual refined boundary — the
    # original input edges may have been split by Steiner points, so drawing
    # them as single lines would cut through the interior.
    # Fall back to the --constraints .txt file only when no embedded edges exist.
    constraint_edges_to_draw = []  # list of (v1_idx, v2_idx) into verts array

    if len(fixed_edges) > 0:
        # Already in CDT internal vertex indices — use directly.
        constraint_edges_to_draw = fixed_edges.tolist()
    elif constraints_path:
        _, c_edges = read_cdt_input(constraints_path)
        # CDT input vertex 0 == OFF vertex 3 (offset by nSuperTriangleVertices)
        offset = 3
        constraint_edges_to_draw = [[e[0] + offset, e[1] + offset]
                                     for e in c_edges]

    if constraint_edges_to_draw:
        for e in constraint_edges_to_draw:
            v1, v2 = e[0], e[1]
            ax.plot([verts[v1, 0], verts[v2, 0]],
                    [verts[v1, 1], verts[v2, 1]],
                    color='crimson', linewidth=1.2, zorder=3)
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color='steelblue', linewidth=0.8, label='Delaunay edge'),
            Line2D([0], [0], color='crimson',   linewidth=1.2, label='Constraint edge'),
        ], loc='upper right')

    # --- Steiner / user vertices ---
    domain_vert_indices = sorted(set(domain_tris.flatten()))
    dv = verts[domain_vert_indices]
    ax.scatter(dv[:, 0], dv[:, 1], s=4, color='steelblue', zorder=4)

    # Clamp axes to the domain bounding box so the super-triangle vertices
    # (which are at extreme coordinates) don't shrink the visible region.
    pad = max(float(np.ptp(dv[:, 0])), float(np.ptp(dv[:, 1]))) * 0.04
    ax.set_xlim(dv[:, 0].min() - pad, dv[:, 0].max() + pad)
    ax.set_ylim(dv[:, 1].min() - pad, dv[:, 1].max() + pad)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a CDT/Ruppert mesh stored in OFF format.")
    parser.add_argument("off_file",
                        help="Path to the OFF file produced by the C++ CDT code")
    parser.add_argument("--constraints", metavar="TXT",
                        help="CDT input .txt file for constraint edge overlay")
    parser.add_argument("--save", metavar="IMAGE",
                        help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    plot(args.off_file,
         constraints_path=args.constraints,
         save_path=args.save)


if __name__ == "__main__":
    main()
