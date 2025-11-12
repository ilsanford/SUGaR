import ROOT as M
import numpy as np
import argparse
import os
import sys
from VertexIDAndPlotting import VertexFinder

# Command line arguments
parser = argparse.ArgumentParser(description="Compute azimuthal angle phi in photon frame for all pair events")
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
GeometryName = "../../MEGAlib_Data/Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
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
Clusterizer.SetParameters(1, -1)
Clusterizer.PreAnalysis()

vertex_finder = VertexFinder(Geometry, NumberOfLayers=2)

# Calculating phi
def ComputePhi_RandomPol(vtx, photon_dir, pol_vec, hit1, hit2):
    k = np.asarray(photon_dir) 
    if np.linalg.norm(k) == 0:
        return None

    k /= np.linalg.norm(k)
    
    # Ensure polarization is perpendicular
    e_pol = np.asarray(pol_vec, dtype=float)
    e_pol -= np.dot(e_pol, k) * k
    if np.linalg.norm(e_pol) == 0: # to handle unpolarized case
        # pick arbitrary vector perpendicular to k
        tmp = np.array([1.,0.,0.])  # not sure how valid this is
        if abs(np.dot(tmp,k)) > 0.9: tmp = np.array([0.,1.,0.])
        e_pol = tmp - np.dot(tmp,k)*k
    e_pol /= np.linalg.norm(e_pol)
    
    e1 = e_pol
    e2 = np.cross(k, e1)
    e2 /= np.linalg.norm(e2)
    
    # hits
    pos1 = np.array([hit1.GetPosition().X(), hit1.GetPosition().Y(), hit1.GetPosition().Z()])
    pos2 = np.array([hit2.GetPosition().X(), hit2.GetPosition().Y(), hit2.GetPosition().Z()])
    vpos = vtx.GetPosition() # slightly redundant
    
    d1 = pos1 - vpos
    d2 = pos2 - vpos
    if np.linalg.norm(d1)==0 or np.linalg.norm(d2)==0:
        return None
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    
    
    # project into photon frame 
    phi1 = np.arctan2(np.dot(d1, e2), np.dot(d1, e1)) + 2*np.pi
    phi2 = np.arctan2(np.dot(d2, e2), np.dot(d2, e1)) + 2*np.pi

    # bisector
    phi = (phi1 + phi2) / 2
    if abs(phi1 - phi2) > np.pi:
        phi += np.pi

    # wrap to [-pi, pi]
    return np.arctan2(np.sin(phi), np.cos(phi))
    
    '''
    # pair plane
    n_plane = np.cross(d1, d2)
    if np.linalg.norm(n_plane)==0:
        return None
    n_plane /= np.linalg.norm(n_plane)
    
    n_proj = n_plane - np.dot(n_plane,k)*k
    if np.linalg.norm(n_proj)==0:
        return None
    n_proj /= np.linalg.norm(n_proj)
    
    phi = np.arctan2(np.dot(n_proj,e2), np.dot(n_proj,e1))
    return np.arctan2(np.sin(phi), np.cos(phi))
    '''

# Event loop
phi_values = []
total_events = 0

# Reader to look at Evta events as well
EvtaReader = M.MFileEventsEvta(Geometry)
EvtaReader.Open(M.MString(sim_file))

while True:
    sim_event = SimReader.GetNextEvent()
    # print(dir(sim_event))
    if not sim_event:
        break
    
    total_events += 1
    photon_vec = sim_event.GetIAAt(0).GetDirection()

    photon_IA = sim_event.GetIAAt(0) 
    theta, phi = photon_vec.Theta(), photon_vec.Phi()

    photon_dir = np.array([
        photon_IA.GetDirection().X(), # i think this is right?
        photon_IA.GetDirection().Y(),
        photon_IA.GetDirection().Z()
    ])


    try:
        pol_vec = np.array([
            photon_IA.GetPolarisation().X(),
            photon_IA.GetPolarisation().Y(),
            photon_IA.GetPolarisation().Z()
        ])
    except AttributeError:
        pol_vec = np.array([0., 0., 0.])
    
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
    
    vertices = vertex_finder.FindVertices(ClusteredRE, theta=theta, phi=phi)
    # print(f"Number of vertices found: {len(vertices)}")

    # Compute phi
    for vtx in vertices:
        hits = [h for h in vtx.AllRESEs if h.GetEnergy() > 0]
        if len(hits) < 2:
            continue
        hit1, hit2 = hits[:2]
        phi_val = ComputePhi_RandomPol(vtx, photon_dir, pol_vec, hit1, hit2)
        if phi_val is not None:
            phi_values.append(phi_val)

SimReader.Close()
EvtaReader.Close()

# Save results
out_file = os.path.splitext(os.path.basename(sim_file))[0] + "_PhiValues.txt"
np.savetxt(out_file, phi_values)
print(f"Wrote {len(phi_values)} phi values to {out_file}")
print(f"Total events: {total_events}")

# look at whether or not hits could be misclassified based on the distance requirement of 5cm (imagine it picks hits from same track)