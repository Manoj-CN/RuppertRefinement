/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Save a CDT triangulation to an OFF file readable by visualize_mesh.py.
 */

#ifndef CDT_SaveToOff_h
#define CDT_SaveToOff_h

#include <CDT.h>

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace CDT
{

/**
 * Save the triangulation (all vertices and triangles, including the
 * super-triangle) to an Object File Format (OFF) file.
 *
 * The file can be visualized with:
 *   python visualize_mesh.py output.off
 *   python visualize_mesh.py output.off --constraints inputs/island.txt
 *
 * @param path  destination file path
 * @param cdt   triangulation to save (call eraseOuterTriangles() first if
 *              you want only domain triangles in the file)
 */
template <typename T, typename TNearPointLocator>
void saveToOff(
    const std::string& path,
    const Triangulation<T, TNearPointLocator>& cdt)
{
    std::ofstream f(path);
    if(!f.is_open())
        throw std::runtime_error("saveToOff: cannot open '" + path + "'");

    f.precision(std::numeric_limits<T>::digits10 + 1);
    f << "OFF\n";
    f << cdt.vertices.size() << ' ' << cdt.triangles.size() << " 0\n";

    for(const auto& v : cdt.vertices)
        f << v.x << ' ' << v.y << " 0\n";

    for(const auto& t : cdt.triangles)
        f << "3 " << t.vertices[0] << ' '
          << t.vertices[1] << ' ' << t.vertices[2] << '\n';

    // Append the current fixed (constraint) edges as comment lines so that
    // visualize_mesh.py can draw the actual refined boundary, not the original
    // pre-refinement edges (which may have been split by Steiner points).
    // Format:  # FIXED_EDGES <count>
    //          # <v1> <v2>
    //          ...
    f << "# FIXED_EDGES " << cdt.fixedEdges.size() << '\n';
    for(const auto& e : cdt.fixedEdges)
        f << "# " << e.v1() << ' ' << e.v2() << '\n';
}

} // namespace CDT

#endif // CDT_SaveToOff_h
