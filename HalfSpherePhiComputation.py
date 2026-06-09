import ROOT as M
import numpy as np
import argparse
import os
import sys
from VertexIDAndPlottingNEW import VertexFinder

# Command line arguments
parser = argparse.ArgumentParser(description="Compute azimuthal angle phi in reference to RelativeX for all pair events")
parser.add_argument("sim_file", help="Path to the MEGAlib .sim or .sim.gz file")
args = parser.parse_args()

sim_file = args.sim_file
if not os.path.exists(sim_file):
    raise FileNotFoundError(f"Simulation file not found: {sim_file}")

# Load MEGAlib
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
G = M.MGlobal()
G.Initialize()

# Load geometry
GeometryName = "/Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
Geometry = M.MGeometryRevan()
if Geometry.ScanSetupFile(M.MString(GeometryName)):
    print("Geometry " + GeometryName + " loaded!")
else:
    print("Unable to load geometry " + GeometryName + " - Aborting!")
    sys.exit(1)

# Prepare SIM reader and clusterizer -> Open with Sim to get MC information
SimReader = M.MFileEventsSim(Geometry)
SimReader.Open(M.MString(sim_file))

Clusterizer = M.MERHitClusterizer()
Clusterizer.SetGeometry(Geometry)
Clusterizer.SetParameters(0,0,0,0,0,0,0,0,True) # no clustering
#Clusterizer.SetParameters(1, -1) 
Clusterizer.PreAnalysis()

vertex_finder = VertexFinder(Geometry, NumberOfLayers=0)

# Calculating phi
def ComputePhi_RelativeX(vtx, photon_dir, hit1, hit2):
    k = np.asarray(photon_dir)
    if np.linalg.norm(k) == 0:
        return None
    k /= np.linalg.norm(k)

    x_hat = np.array([1.0, 0.0, 0.0]) # in detector frame
    e_pol = np.cross(k, x_hat) # this gives the RelativeX direction defined in MEGAlib
    e_pol /= np.linalg.norm(e_pol)

    e2 = np.cross(k, e_pol)  # creates an orthonormal basis for a plane perpendicular to the photon direction
    e2 /= np.linalg.norm(e2)

    # hit positions
    pos1 = np.array([hit1.GetPosition().X(),hit1.GetPosition().Y(),hit1.GetPosition().Z()])
    pos2 = np.array([hit2.GetPosition().X(),hit2.GetPosition().Y(),hit2.GetPosition().Z()])
    vpos = np.array([vtx.GetXPosition(),vtx.GetYPosition(),vtx.GetZPosition()])

    # direction vectors for the electron/positron hits relative to the identified vertex location
    d1 = pos1 - vpos
    d2 = pos2 - vpos
    if np.linalg.norm(d1) == 0 or np.linalg.norm(d2) == 0:
        return None
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)

    # project d1 and d2 into transverse plane
    d1_perp = d1 - np.dot(d1, k) * k
    d2_perp = d2 - np.dot(d2, k) * k
    if np.linalg.norm(d1_perp) == 0 or np.linalg.norm(d2_perp) == 0:
        return None
    d1_perp /= np.linalg.norm(d1_perp)
    d2_perp /= np.linalg.norm(d2_perp)

    # azimuthal angle of each track in transverse plane
    phi1 = np.arctan2(np.dot(d1_perp, e2), np.dot(d1_perp, e_pol))
    phi2 = np.arctan2(np.dot(d2_perp, e2), np.dot(d2_perp, e_pol))

    # bisector
    phi = (phi1 + phi2) / 2
    if abs(phi1 - phi2) > np.pi:
        phi += np.pi

    return np.arctan2(np.sin(phi), np.cos(phi))
    '''
    # plane normal to the d1 d2 plane
    n_plane = np.cross(d1, d2)
    n_plane /= np.linalg.norm(n_plane)

    n_proj = n_plane - np.dot(n_plane, k) * k # remove the component of k that lies in n_plane
    n_proj /= np.linalg.norm(n_proj)

    phi = np.arctan2(np.dot(n_proj, e2), np.dot(n_proj, e_pol))
    return np.arctan2(np.sin(phi), np.cos(phi)) # wrapping to [-pi, pi]
    '''

# Event loop
phi_values = []
total_events = 0

# Reader to look at Evta events as well
EvtaReader = M.MFileEventsEvta(Geometry)
EvtaReader.Open(M.MString(sim_file))

n_vertices = 0
n_type1ht1ht = 0

while True:
    sim_event = SimReader.GetNextEvent()
    # print(dir(sim_event))
    if not sim_event:
        break
    
    M.SetOwnership(sim_event, True)

    total_events += 1
    photon_vec = sim_event.GetIAAt(0).GetDirection()

    photon_IA = sim_event.GetIAAt(0) 
    theta, phi = photon_vec.Theta(), photon_vec.Phi()

    photon_dir = np.array([
        photon_IA.GetDirection().X(),
        photon_IA.GetDirection().Y(),
        photon_IA.GetDirection().Z()
    ])
    
    # Need to open with Evta to cluster and such
    RE = EvtaReader.GetNextEvent()
    if not RE:
        break
    M.SetOwnership(RE, True)
    if not RE or not RE.IsValid():
        continue

    REI = M.MRawEventIncarnations()
    M.SetOwnership(REI, True)
    REI.SetInitialRawEvent(RE)
    Clusterizer.Analyze(REI)
    ClusteredRE = REI.GetInitialRawEvent()
    M.SetOwnership(ClusteredRE, True)
    
    vertices = vertex_finder.FindVertices(ClusteredRE, theta=theta, phi=phi, polarization=True)
    # print(f"Number of vertices found: {len(vertices)}")

    for vtx in vertices:
        n_vertices += 1
        hits = [h for h in vtx.AllRESEs if h.GetEnergy() > 0]

        if vtx.vertex_type == 'type_1ht1ht':
            n_type1ht1ht += 1
            continue
        if len(hits) < 2:
            n_toofew_hits += 1
            continue

        hit1, hit2 = hits[:2]
        phi_val = ComputePhi_RelativeX(vtx, photon_dir, hit1, hit2)
        if phi_val is None:
            n_none_phi += 1
            continue
        phi_values.append(phi_val)
        

SimReader.Close()
EvtaReader.Close()

# Save results
out_file = os.path.splitext(os.path.basename(sim_file))[0] + "_PhiValues.txt"
np.savetxt(out_file, phi_values)
print(f"Wrote {len(phi_values)} phi values to {out_file}")
print(f"Total events: {total_events}")

# print after event loop
print(f"Total vertices found: {n_vertices}")
print(f"Removed type_1ht1ht: {n_type1ht1ht}")
print(f"Good phi values: {len(phi_values)}")

