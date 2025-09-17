import ROOT as M
import argparse

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
    quit()

# Reading in the sim file
parser = argparse.ArgumentParser()
parser.add_argument("simfile")
args = parser.parse_args()
simfile = args.simfile

# Setting up a dictionary to store the counts in each material
counts_in_material = {}

Reader = M.MFileEventsSim(Geometry)
if not Reader.Open(M.MString(simfile)):
    print(f"Failed to open {simfile}"); quit()

while (Event := Reader.GetNextEvent()):
    M.SetOwnership(Event, True)
    for i in range(Event.GetNIAs()):
        IA = Event.GetIAAt(i)
        # Selecting only the pair processes in the sim file
        if IA.GetProcess() != "PAIR":
            continue
        
        # Extracting the poistion of the pair conversion and its associated volume
        pos = IA.GetPosition()            
        volume = Geometry.GetVolume(pos)
        # For the volume, determine the material and increase the count
        if volume:
            mat = volume.GetMaterial()
            if mat:
                material_name = mat.GetName().ToString()
                counts_in_material[material_name] = counts_in_material.get(material_name, 0) + 1
                
Reader.Close()

print("---- Pair Conversions by Material ----")
total = sum(counts_in_material.values())
print(f"Total conversions: {total}")
for mat, count in sorted(counts_in_material.items(), key=lambda x: -x[1]):
    print(f"{mat}: {count}")
