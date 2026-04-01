/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Ruppert's Delaunay refinement algorithm
 *
 * Improves triangle quality in a Constrained Delaunay Triangulation by
 * inserting Steiner points (circumcenters of bad triangles). Constraint
 * edges that would be encroached are split at their midpoint first.
 *
 * Reference:
 *   J. Ruppert, "A Delaunay Refinement Algorithm for Quality 2-Dimensional
 *   Mesh Generation", Journal of Algorithms, 18(3):548-585, 1995.
 */

#ifndef CDT_RuppertRefinement_h
#define CDT_RuppertRefinement_h

#include <CDT.h>
#include <CDTUtils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace CDT
{

namespace ruppert_detail
{

/// Compute circumcenter of triangle (a, b, c)
template <typename T>
V2d<T> circumcenter(const V2d<T>& a, const V2d<T>& b, const V2d<T>& c)
{
    const T ax = a.x, ay = a.y;
    const T bx = b.x, by = b.y;
    const T cx = c.x, cy = c.y;
    const T D = T(2) * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    const T a2 = ax * ax + ay * ay;
    const T b2 = bx * bx + by * by;
    const T c2 = cx * cx + cy * cy;
    return V2d<T>(
        (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D,
        (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D);
}

/// Squared circumradius of triangle (a, b, c)
template <typename T>
T circumradiusSq(const V2d<T>& a, const V2d<T>& b, const V2d<T>& c)
{
    const V2d<T> cc = circumcenter(a, b, c);
    return distanceSquared(cc, a);
}

/// Squared length of shortest edge of triangle (a, b, c)
template <typename T>
T shortestEdgeSq(const V2d<T>& a, const V2d<T>& b, const V2d<T>& c)
{
    return std::min(
        std::min(distanceSquared(a, b), distanceSquared(b, c)),
        distanceSquared(c, a));
}

/// Squared length of longest edge of triangle (a, b, c)
template <typename T>
T longestEdgeSq(const V2d<T>& a, const V2d<T>& b, const V2d<T>& c)
{
    return std::max(
        std::max(distanceSquared(a, b), distanceSquared(b, c)),
        distanceSquared(c, a));
}

/**
 * Test whether point p encroaches on the diametral circle of segment (a, b).
 *
 * A point encroaches if it lies strictly inside the circle whose diameter
 * is the segment, i.e. the angle subtended at p is obtuse:
 *   dot(a - p, b - p) < 0
 */
template <typename T>
bool encroaches(const V2d<T>& p, const V2d<T>& a, const V2d<T>& b)
{
    return (a.x - p.x) * (b.x - p.x) + (a.y - p.y) * (b.y - p.y) < T(0);
}

/// Return true if v has finite (non-NaN, non-Inf) coordinates
template <typename T>
bool isFinite(const V2d<T>& v)
{
    return v.x == v.x && v.y == v.y && // NaN check
           v.x != std::numeric_limits<T>::infinity() &&
           v.x != -std::numeric_limits<T>::infinity() &&
           v.y != std::numeric_limits<T>::infinity() &&
           v.y != -std::numeric_limits<T>::infinity();
}

/**
 * Test whether triangle (a, b, c) is "bad".
 *
 * A triangle is bad if:
 *   - its circumradius-to-shortest-edge ratio exceeds the quality bound B,
 *     where B = 1 / (2 * sin(minAngle)), OR
 *   - its longest edge exceeds maxEdgeLenSq (when maxEdgeLenSq > 0).
 *
 * The ratio test is equivalent to: min angle < minAngleDegrees.
 */
template <typename T>
bool isBadTriangle(
    const V2d<T>& a,
    const V2d<T>& b,
    const V2d<T>& c,
    T qualityBoundSq,
    T maxEdgeLenSq)
{
    const T lmin2 = shortestEdgeSq(a, b, c);
    // Skip degenerate triangles
    if(lmin2 < std::numeric_limits<T>::epsilon())
        return false;
    const T R2 = circumradiusSq(a, b, c);
    if(R2 / lmin2 > qualityBoundSq)
        return true;
    if(maxEdgeLenSq > T(0) && longestEdgeSq(a, b, c) > maxEdgeLenSq)
        return true;
    return false;
}

/**
 * Insert v into each vector in 'vecs' if not already present.
 * Replicates internal detail::insert_unique for use with public members.
 */
template <typename T>
void insertUnique(EdgeVec& vec, const T& val)
{
    if(std::find(vec.begin(), vec.end(), val) == vec.end())
        vec.push_back(val);
}

/**
 * Split a fixed (constraint) edge at a newly inserted vertex.
 *
 * Replicates the logic of the private Triangulation::splitFixedEdge using
 * only publicly accessible members (fixedEdges, overlapCount, pieceToOriginals).
 *
 * @param cdt   the triangulation
 * @param edge  the fixed edge to split
 * @param iMid  index of the already-inserted midpoint vertex
 */
template <typename T, typename TNearPointLocator>
void splitFixedEdgePublic(
    Triangulation<T, TNearPointLocator>& cdt,
    const Edge& edge,
    VertInd iMid)
{
    const Edge half1(edge.v1(), iMid);
    const Edge half2(iMid, edge.v2());

    // Update fixed-edge set
    cdt.fixedEdges.erase(edge);
    if(!cdt.fixedEdges.insert(half1).second)
        ++cdt.overlapCount[half1];
    if(!cdt.fixedEdges.insert(half2).second)
        ++cdt.overlapCount[half2];

    // Propagate overlap count to halves
    typedef unordered_map<Edge, BoundaryOverlapCount>::const_iterator OlapIt;
    const OlapIt olapIt = cdt.overlapCount.find(edge);
    if(olapIt != cdt.overlapCount.end())
    {
        cdt.overlapCount[half1] += olapIt->second;
        cdt.overlapCount[half2] += olapIt->second;
        cdt.overlapCount.erase(olapIt);
    }

    // Propagate piece-to-original mapping
    EdgeVec newOriginals(1, edge);
    typedef unordered_map<Edge, EdgeVec>::const_iterator PieceIt;
    const PieceIt pieceIt = cdt.pieceToOriginals.find(edge);
    if(pieceIt != cdt.pieceToOriginals.end())
    {
        newOriginals = pieceIt->second;
        cdt.pieceToOriginals.erase(pieceIt);
    }
    EdgeVec& origHalf1 = cdt.pieceToOriginals[half1];
    EdgeVec& origHalf2 = cdt.pieceToOriginals[half2];
    typedef EdgeVec::const_iterator EVIt;
    for(EVIt it = newOriginals.begin(); it != newOriginals.end(); ++it)
    {
        insertUnique(origHalf1, *it);
        insertUnique(origHalf2, *it);
    }
}

} // namespace ruppert_detail

/**
 * Refine a Constrained Delaunay Triangulation using Ruppert's algorithm.
 *
 * Inserts Steiner points to eliminate triangles whose minimum angle is below
 * @p minAngleDegrees. The circumcenter of each bad triangle is inserted
 * unless it would encroach on a constraint edge, in which case that
 * constraint edge is split at its midpoint first.
 *
 * @note Call this after insertVertices() and insertEdges() / conformToEdges()
 *       but <b>before</b> any erase...() method.
 * @note Termination is guaranteed for minAngleDegrees &le; 20.7&deg;.
 *       For larger values the algorithm may not terminate; use
 *       @p maxSteinerPoints as a safety limit.
 * @note When constraint edges are present, set @p minDepth = 1 (the default)
 *       so that only triangles inside the outermost constraint boundary are
 *       refined. Triangles outside the boundary (between the boundary and the
 *       super-triangle) are skipped, which avoids wasting Steiner points on
 *       regions that will be erased later. Pass 0 to refine everything.
 *
 * @tparam T                 vertex coordinate type (float or double)
 * @tparam TNearPointLocator near-point locator (default: LocatorKDTree<T>)
 * @param cdt                triangulation to refine (modified in-place)
 * @param minAngleDegrees    minimum allowed triangle angle (default: 20.0)
 * @param maxEdgeLength      maximum allowed edge length, 0 = no limit
 * @param maxSteinerPoints   Steiner point insertion limit, 0 = no limit
 * @param minDepth           minimum triangle layer depth to refine (default: 1)
 */
template <typename T, typename TNearPointLocator>
void refineRuppert(
    Triangulation<T, TNearPointLocator>& cdt,
    T minAngleDegrees = T(20),
    T maxEdgeLength = T(0),
    std::size_t maxSteinerPoints = 0,
    LayerDepth minDepth = 1)
{
    // Precompute quality bound: B = 1 / (2 * sin(minAngle))
    // Triangle is bad when R / lmin > B, i.e. R^2 / lmin^2 > B^2
    const T pi = T(3.14159265358979323846);
    const T minAngleRad = minAngleDegrees * pi / T(180);
    const T sinMin = std::sin(minAngleRad);
    const T B = (sinMin > T(0)) ? T(1) / (T(2) * sinMin)
                                : std::numeric_limits<T>::max();
    const T qualityBoundSq = B * B;
    const T maxEdgeLenSq =
        (maxEdgeLength > T(0)) ? maxEdgeLength * maxEdgeLength : T(0);

    // Whether to use layer-depth filtering (only meaningful when there are
    // constraint edges; without them every triangle is at depth 0).
    const bool useDepthFilter =
        (minDepth > LayerDepth(0)) && !cdt.fixedEdges.empty();

    // Total vertex insertions (circumcenters + constraint-edge midpoints).
    // Both types consume the maxSteinerPoints budget so that an encroachment
    // cascade of constraint-edge splits cannot bypass the limit.
    std::size_t insertionCount = 0;
    bool anyChange = true;

    while(anyChange)
    {
        anyChange = false;

        // Recompute triangle depths at the start of each pass so newly
        // inserted triangles are classified correctly.
        std::vector<LayerDepth> triDepths;
        if(useDepthFilter)
            triDepths = cdt.calculateTriangleDepths();

        for(TriInd iT(0); iT < TriInd(cdt.triangles.size()); ++iT)
        {
            if(maxSteinerPoints > 0 && insertionCount >= maxSteinerPoints)
                return;

            const Triangle& tri = cdt.triangles[iT];

            // Skip triangles touching the super-triangle
            if(touchesSuperTriangle(tri))
                continue;

            // Skip triangles outside the constraint domain
            if(useDepthFilter && triDepths[iT] < minDepth)
                continue;

            const V2d<T>& a = cdt.vertices[tri.vertices[0]];
            const V2d<T>& b = cdt.vertices[tri.vertices[1]];
            const V2d<T>& c = cdt.vertices[tri.vertices[2]];

            if(!ruppert_detail::isBadTriangle(
                   a, b, c, qualityBoundSq, maxEdgeLenSq))
                continue;

            // Compute circumcenter of the bad triangle
            const V2d<T> cc = ruppert_detail::circumcenter(a, b, c);

            // Skip if circumcenter is non-finite (degenerate triangle)
            if(!ruppert_detail::isFinite(cc))
                continue;

            // Check whether the circumcenter encroaches on any fixed edge
            bool encroached = false;
            Edge encroachedEdge(VertInd(0), VertInd(1)); // initialised below
            for(EdgeUSet::const_iterator eIt = cdt.fixedEdges.begin();
                eIt != cdt.fixedEdges.end();
                ++eIt)
            {
                const V2d<T>& ea = cdt.vertices[eIt->v1()];
                const V2d<T>& eb = cdt.vertices[eIt->v2()];
                if(ruppert_detail::encroaches(cc, ea, eb))
                {
                    encroachedEdge = *eIt;
                    encroached = true;
                    break;
                }
            }

            if(encroached)
            {
                // Split the encroached constraint edge at its midpoint.
                //
                // Keep the edge in fixedEdges during insertion so that:
                //  (a) insertVertexOnEdge(doHandleFixedSplitEdge=true) detects
                //      it and calls splitFixedEdge internally, correctly
                //      updating fixedEdges/overlapCount/pieceToOriginals; and
                //  (b) Delaunay legalization does not flip the fixed edge.
                const V2d<T>& ea = cdt.vertices[encroachedEdge.v1()];
                const V2d<T>& eb = cdt.vertices[encroachedEdge.v2()];
                const V2d<T> mid((ea.x + eb.x) / T(2), (ea.y + eb.y) / T(2));

                const VertInd iMid =
                    static_cast<VertInd>(cdt.vertices.size());

                std::vector<V2d<T> > newVerts(1, mid);
                cdt.insertVertices(newVerts);
                ++insertionCount;

                // Normally insertVertexOnEdge calls splitFixedEdge internally.
                // If the midpoint landed inside a triangle instead of on the
                // edge (a rare floating-point edge case), update fixedEdges
                // manually so the triangulation remains in a consistent state.
                if(cdt.fixedEdges.count(encroachedEdge))
                {
                    ruppert_detail::splitFixedEdgePublic(
                        cdt, encroachedEdge, iMid);
                }
            }
            else
            {
                // Insert the circumcenter as a new Steiner point.
                // Guard against circumcenters that fall outside the
                // super-triangle (can happen for very flat triangles near
                // the domain boundary when there are no constraints).
                // If insertion fails we simply leave the triangle as-is and
                // continue looking for other bad triangles.
#ifdef CDT_CXX11_IS_SUPPORTED
                try
                {
                    std::vector<V2d<T> > newVerts(1, cc);
                    cdt.insertVertices(newVerts);
                    ++insertionCount;
                }
                catch(const std::exception&)
                {
                    continue; // skip this triangle, try the next bad one
                }
#else
                std::vector<V2d<T> > newVerts(1, cc);
                cdt.insertVertices(newVerts);
                ++insertionCount;
#endif
            }

            anyChange = true;
            // Restart the scan: the triangulation was modified (indices may
            // have been invalidated by reallocation).
            break;
        }
    }
}

} // namespace CDT

#endif // CDT_RuppertRefinement_h
