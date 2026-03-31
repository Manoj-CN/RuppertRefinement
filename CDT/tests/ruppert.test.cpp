#include <CDT.h>
#include <RuppertRefinement.h>
#include <SaveToOff.h>
#include <VerifyTopology.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace CDT;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

using CoordTypes = std::tuple<float, double>;

template <class T>
using Vertices = std::vector<V2d<T>>;

/// Load a CDT input file (same format used by existing tests).
/// First line: nVerts nEdges
/// Then nVerts lines "x y", then nEdges lines "v1 v2".
template <typename T>
std::pair<Vertices<T>, std::vector<Edge>>
readInput(const std::string& path)
{
    std::ifstream f(path);
    REQUIRE(f.is_open());

    IndexSizeType nVerts, nEdges;
    f >> nVerts >> nEdges;

    Vertices<T> vv;
    vv.reserve(nVerts);
    for(IndexSizeType i = 0; i < nVerts; ++i)
    {
        T x, y;
        f >> x >> y;
        vv.push_back(V2d<T>(x, y));
    }

    std::vector<Edge> ee;
    ee.reserve(nEdges);
    for(IndexSizeType i = 0; i < nEdges; ++i)
    {
        VertInd v1, v2;
        f >> v1 >> v2;
        ee.push_back(Edge(v1, v2));
    }
    return {vv, ee};
}

/// Minimum angle (degrees) of triangle with vertices a, b, c.
template <typename T>
T minAngleDeg(const V2d<T>& a, const V2d<T>& b, const V2d<T>& c)
{
    // Law of cosines: cos(A) = (b²+c²-a²)/(2bc)
    const T ab = distance(a, b);
    const T bc = distance(b, c);
    const T ca = distance(c, a);
    if(ab < std::numeric_limits<T>::epsilon() ||
       bc < std::numeric_limits<T>::epsilon() ||
       ca < std::numeric_limits<T>::epsilon())
        return T(0); // degenerate

    const T pi = T(3.14159265358979323846);
    const T rad2deg = T(180) / pi;
    const T cosA =
        std::max(T(-1), std::min(T(1), (ab*ab + ca*ca - bc*bc) / (T(2)*ab*ca)));
    const T cosB =
        std::max(T(-1), std::min(T(1), (ab*ab + bc*bc - ca*ca) / (T(2)*ab*bc)));
    const T cosC =
        std::max(T(-1), std::min(T(1), (bc*bc + ca*ca - ab*ab) / (T(2)*bc*ca)));

    return std::min({std::acos(cosA), std::acos(cosB), std::acos(cosC)}) *
           rad2deg;
}

/// Returns the minimum angle across all non-super-triangle triangles.
template <typename T, typename TNearPointLocator>
T meshMinAngleDeg(const Triangulation<T, TNearPointLocator>& cdt)
{
    T minAngle = std::numeric_limits<T>::max();
    for(const Triangle& tri : cdt.triangles)
    {
        if(touchesSuperTriangle(tri))
            continue;
        const V2d<T>& a = cdt.vertices[tri.vertices[0]];
        const V2d<T>& b = cdt.vertices[tri.vertices[1]];
        const V2d<T>& c = cdt.vertices[tri.vertices[2]];
        minAngle = std::min(minAngle, minAngleDeg(a, b, c));
    }
    return minAngle;
}

/// Count non-super-triangle triangles.
template <typename T, typename TNearPointLocator>
std::size_t domainTriCount(const Triangulation<T, TNearPointLocator>& cdt)
{
    std::size_t n = 0;
    for(const Triangle& tri : cdt.triangles)
        if(!touchesSuperTriangle(tri))
            ++n;
    return n;
}

/// Save an OFF file for a test.  Files land in the working directory
/// (CDT/tests/ when run via CTest / the test binary).
/// Name pattern: ruppert_<base>_<before|after>_<float|double>.off
///
/// For the "before" stage a copy of the CDT is made and outer triangles are
/// erased before saving so the file contains only the inner domain triangles
/// (matching the "after" view).  This avoids showing the outer convex-hull
/// triangles that are present in the CDT before refinement.
template <typename T, typename TNearPointLocator>
void saveTestOff(
    const Triangulation<T, TNearPointLocator>& cdt,
    const std::string& base,
    const std::string& stage) // "before" or "after"
{
    const std::string typeName =
        std::is_same<T, float>::value ? "float" : "double";
    const std::string path =
        "ruppert_" + base + "_" + stage + "_" + typeName + ".off";

    if(stage == "before")
    {
        // Copy so the original CDT is not modified
        Triangulation<T, TNearPointLocator> copy = cdt;
        if(!copy.fixedEdges.empty())
            copy.eraseOuterTriangles();
        else
            copy.eraseSuperTriangle();
        CDT::saveToOff(path, copy);
    }
    else
    {
        CDT::saveToOff(path, cdt);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Helper: build a CDT from hand-coded vertices, refine, check quality.
TEMPLATE_LIST_TEST_CASE(
    "Ruppert - four corner square, no constraints",
    "[Ruppert]",
    CoordTypes)
{
    Triangulation<TestType> cdt;
    cdt.insertVertices(Vertices<TestType>{
        {TestType(0), TestType(0)},
        {TestType(1), TestType(0)},
        {TestType(1), TestType(1)},
        {TestType(0), TestType(1)}});
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "four_corner_square", "before");

    const std::size_t trisBefore = domainTriCount(cdt);
    refineRuppert(cdt, TestType(20));
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "four_corner_square", "after");

    REQUIRE(domainTriCount(cdt) >= trisBefore);
    REQUIRE(meshMinAngleDeg(cdt) >= TestType(19.5));
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - unit square with boundary constraints",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/unit square.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "unit_square", "before");

    const std::size_t trisBefore = domainTriCount(cdt);
    const TestType minAngle = TestType(20);
    refineRuppert(cdt, minAngle, TestType(0), /*maxSteiner=*/5000);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "unit_square", "after");

    REQUIRE(domainTriCount(cdt) >= trisBefore);
    REQUIRE(meshMinAngleDeg(cdt) >= TestType(19.5));
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - island polygon",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/island.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "island", "before");

    const TestType minAngle = TestType(15);
    refineRuppert(cdt, minAngle, TestType(0), /*maxSteiner=*/10000);
    REQUIRE(verifyTopology(cdt));

    cdt.eraseOuterTriangles();
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "island", "after");

    REQUIRE(meshMinAngleDeg(cdt) >= TestType(14.5));
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - kidney polygon",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/kidney.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "kidney", "before");

    const TestType minAngle = TestType(15);
    refineRuppert(cdt, minAngle, TestType(0), /*maxSteiner=*/5000);
    REQUIRE(verifyTopology(cdt));

    cdt.eraseOuterTriangles();
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "kidney", "after");

    REQUIRE(meshMinAngleDeg(cdt) >= TestType(14.5));
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - Capital A (polygon with hole)",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/Capital A.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "capital_a", "before");

    const std::size_t vertsBefore = cdt.vertices.size();
    refineRuppert(cdt, TestType(20), TestType(0), /*maxSteiner=*/5000);
    REQUIRE(verifyTopology(cdt));
    REQUIRE(cdt.vertices.size() > vertsBefore);

    cdt.eraseOuterTrianglesAndHoles();
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "capital_a", "after");
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - guitar outline",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/guitar no box.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "guitar", "before");

    const TestType minAngle = TestType(15);
    refineRuppert(cdt, minAngle, TestType(0), /*maxSteiner=*/10000);
    REQUIRE(verifyTopology(cdt));

    cdt.eraseOuterTriangles();
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "guitar", "after");

    REQUIRE(meshMinAngleDeg(cdt) >= TestType(14.5));
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - Constrained Sweden",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/Constrained Sweden.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "sweden", "before");

    const std::size_t vertsBefore = cdt.vertices.size();
    refineRuppert(cdt, TestType(15), TestType(0), /*maxSteiner=*/200);
    REQUIRE(verifyTopology(cdt));
    REQUIRE(cdt.vertices.size() > vertsBefore);

    cdt.eraseOuterTriangles();
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "sweden", "after");
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - max Steiner points limit is respected",
    "[Ruppert]",
    CoordTypes)
{
    const auto [vv, ee] =
        readInput<TestType>("inputs/island.txt");

    Triangulation<TestType> cdt;
    cdt.insertVertices(vv);
    cdt.insertEdges(ee);
    saveTestOff(cdt, "max_steiner", "before");

    const std::size_t vertsBefore = cdt.vertices.size();
    refineRuppert(cdt, TestType(25), TestType(0), /*maxSteiner=*/10);
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "max_steiner", "after");

    REQUIRE(cdt.vertices.size() <= vertsBefore + 200);
}

TEMPLATE_LIST_TEST_CASE(
    "Ruppert - max edge length constraint",
    "[Ruppert]",
    CoordTypes)
{
    Triangulation<TestType> cdt;
    cdt.insertVertices(Vertices<TestType>{
        {TestType(0), TestType(0)},
        {TestType(4), TestType(0)},
        {TestType(4), TestType(4)},
        {TestType(0), TestType(4)}});
    cdt.insertEdges(std::vector<Edge>{
        Edge(VertInd(0), VertInd(1)),
        Edge(VertInd(1), VertInd(2)),
        Edge(VertInd(2), VertInd(3)),
        Edge(VertInd(3), VertInd(0))});
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "max_edge_length", "before");

    const TestType maxLen = TestType(1.5);
    refineRuppert(cdt, TestType(20), maxLen, /*maxInsertions=*/20000);
    REQUIRE(verifyTopology(cdt));

    cdt.eraseOuterTriangles();
    REQUIRE(verifyTopology(cdt));
    saveTestOff(cdt, "max_edge_length", "after");

    const TestType maxLenSq = maxLen * maxLen * TestType(1.01);
    for(const Triangle& tri : cdt.triangles)
    {
        if(touchesSuperTriangle(tri))
            continue;
        const V2d<TestType>& a = cdt.vertices[tri.vertices[0]];
        const V2d<TestType>& b = cdt.vertices[tri.vertices[1]];
        const V2d<TestType>& c = cdt.vertices[tri.vertices[2]];
        REQUIRE(distanceSquared(a, b) <= maxLenSq);
        REQUIRE(distanceSquared(b, c) <= maxLenSq);
        REQUIRE(distanceSquared(c, a) <= maxLenSq);
    }
}
