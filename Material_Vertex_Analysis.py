import ROOT as M
from VertexIDAndPlotting import VertexFinder
from collections import defaultdict
import argparse
import numpy as np
import sys
import os

# Loading MEGAlib
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

# Input event -> material file, sim file, and material
parser = argparse.ArgumentParser(description='Compute phi values for events belonging to a chosen material')
parser.add_argument('--input_file', default=None,
                    help='EventID-> Material text file')
parser.add_argument('--sim_file', default=None,
                    help='Pair-events-only sim file to read')
parser.add_argument('--material', '-m', default='Silicon', help='Material name to process')
args = parser.parse_args()

input_file = args.input_file
sim_file = args.sim_file
target_material = args.material

# Make sure file inputs are valid
if not os.path.exists(input_file):
    print(f"Event ID file not found: {input_file}")
    sys.exit(1)
if not os.path.exists(sim_file):
    print(f"Sim file not found: {sim_file}")
    sys.exit(1)

# Read event IDs grouped by material
event_ids_by_material = defaultdict(set)
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            eid = int(parts[0])
        except ValueError:
            continue
        material = parts[1]
        event_ids_by_material[material].add(eid)

print(f"Material Options: {list(event_ids_by_material.keys())}")
if target_material not in event_ids_by_material:
    print(f"Warning: target material '{target_material}' not found in {input_file}")

# Set of event IDs for the chosen material
target_eids = event_ids_by_material.get(target_material, set())

# Prepare reader and clusterizer
Reader = M.MFileEventsEvta(Geometry)
Reader.Open(M.MString(sim_file))

Clusterizer = M.MERHitClusterizer()
Clusterizer.SetGeometry(Geometry)
Clusterizer.SetParameters(1, -1)
Clusterizer.PreAnalysis()

vertex_finder = VertexFinder(Geometry, NumberOfLayers=2)

# Initialize results
results = {target_material: {"event_ids": [], "phi": []}}

# Loop over the events
while True:
    RE = Reader.GetNextEvent()
    if not RE:
        break
    M.SetOwnership(RE, True)
    if not RE.IsValid():
        continue

    eid = RE.GetEventID()

    # Only process events that belong to the target material
    if eid not in target_eids:
        continue

    # Cluster the event
    REI = M.MRawEventIncarnations()
    M.SetOwnership(REI, True)
    REI.SetInitialRawEvent(RE)
    Clusterizer.Analyze(REI)
    ClusteredRE = REI.GetInitialRawEvent()
    M.SetOwnership(ClusteredRE, True)

    # Find vertices
    vertices = vertex_finder.FindVertices(ClusteredRE, theta=0.0, phi=0.0) # currently hardcoding on-axis simulations -> can change this

    # Compute phi
    for vtx in vertices:
        hit1, hit2 = vtx.AllRESEs
        phi_val = vtx.ComputePhi(theta=0.0, phi=0.0, hit1=hit1, hit2=hit2, ref_direction="RelativeX")
        results[target_material]["event_ids"].append(eid)
        results[target_material]["phi"].append(phi_val)

Reader.Close()

# Construct appropriate output filename
base = os.path.basename(input_file)
name, ext = os.path.splitext(base)
out_phi_file = f"{name}_{target_material}PhiValues.txt"

# Write out phi values for the target material
if len(results[target_material]["phi"]) > 0:
    np.savetxt(out_phi_file, results[target_material]["phi"])
    print(f"Wrote {len(results[target_material]['phi'])} phi values to {out_phi_file}")
else:
    print(f"No phi values computed for material '{target_material}'")

# Print summary
print(f"Material: {target_material}")
print(f"Number of vertices: {len(results[target_material]['event_ids'])}")