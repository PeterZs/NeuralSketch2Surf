"""Geodesic curve tracing utilities for synthetic sketch generation."""
import openmesh as om
import igl
import numpy as np
import random
import sys

def solveEdgeIntersection(d, v, w, P, B, C):
    """Intersect the current geodesic ray with a triangle edge."""
    m11 = np.dot( d, -v)
    m12 = np.dot( v, v)
    m21 = np.dot( d, +w)
    m22 = np.dot( -v, +w)
    rhs1 = np.dot( B-P, -v)
    rhs2 = np.dot( B-P, +w)
    m = np.array([ [m11,m12],[m21,m22]])
    rhs = np.array([rhs1, rhs2]).transpose()
    sol = np.linalg.solve( m, rhs)
    Q1 = P + sol[0] * d
    return sol, Q1


def findNextPoint(mesh, heh, lmbda, theta, P):
    """Advance one segment along the mesh geodesic."""
    if theta > np.pi:
        print(f"Unexpected theta > pi: {theta}")
        heh = mesh.opposite_halfedge_handle(heh)
        theta = theta - np.pi
        lmbda = 1. - lmbda

    he1 = heh
    he2 = mesh.next_halfedge_handle(he1)
    he3 = mesh.next_halfedge_handle(he2)
    Ah = mesh.from_vertex_handle(he1)
    Bh = mesh.from_vertex_handle(he2)
    Ch = mesh.from_vertex_handle(he3)

    A = mesh.point(Ah)
    B = mesh.point(Bh)
    C = mesh.point(Ch)

    u = B - A
    v = C - B
    w = A - C

    n = np.cross( v, w)
    uT = np.cross( n, u)
    # Continue the geodesic direction in the local triangle frame.
    d = np.cos(theta) * u / np.linalg.norm(u) + np.sin(theta) * uT / np.linalg.norm( uT)

    oP = (1.0 - lmbda) * A + lmbda * B
    assert np.allclose( P, oP)

    sol, nP = solveEdgeIntersection( d, v, w, P, B, C)

    if sol[1] < 0. or sol[1] > 1.:
        sol, nP = solveEdgeIntersection( d, -w, +v, P, A, C)

        nHehOpp = he3
        nLmbda = sol[1]
        nU = w
    else:
        if sol[0]<=0.:
            print(f"fatal error: intersection inside segment BC and outside triangle: sol={sol}")
            sys.exit(1)
        nHehOpp = he2
        nLmbda = 1 - sol[1]
        nU = v

    assert sol[1]>=0 and sol[1]<=1., "new intersection should be inside segment"

    nHeh = mesh.opposite_halfedge_handle(nHehOpp)

    nTheta = np.pi - np.arccos( np.dot( d/np.linalg.norm(d), nU/np.linalg.norm(nU)))

    return nHeh, nLmbda, nTheta, nP

def initGeodesic(mesh, v, f, facesWithPoints):
    """Initialize a geodesic from a random or farthest face."""
    if "-farthest" not in sys.argv or facesWithPoints == set():
        face_handles = list(mesh.faces())
        if not face_handles:
            raise ValueError("Mesh has no faces.")
        fh = random.choice(face_handles)
    else:
        vs = np.array([], dtype=int)
        fs = np.array(list(facesWithPoints), dtype=int)
        vt = np.array([], dtype=int)
        ft = np.array(np.arange(mesh.n_faces()))
        d = igl.exact_geodesic(v, f, vs, fs, vt, ft)
        faceIndex = np.argmax(d)
        fh = om.FaceHandle(faceIndex)

    hehs = list(mesh.fh(fh))
    heh = random.choice(hehs)
    from_vh = mesh.from_vertex_handle(heh)
    to_vh = mesh.to_vertex_handle(heh)

    lmbda = random.uniform(0.2, 0.8)

    A = mesh.point(from_vh)
    B = mesh.point(to_vh)

    P = (1.0 - lmbda) * np.array(A) + lmbda * np.array(B)

    theta = random.uniform( np.pi/6., +np.pi - np.pi/6.)

    return fh, heh, lmbda, theta, P
