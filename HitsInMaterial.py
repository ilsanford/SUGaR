import ROOT as M
import argparse
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

# Reading in the sim file
parser = argparse.ArgumentParser(description='Write pair-conversion material info from sim file')
parser.add_argument("simfile", help='Path to the input sim file')
parser.add_argument("--outputfile", default=None, help="Output text file with Event ID and material information (default: <simfile>_MaterialInfo.txt)")
args = parser.parse_args()

simfile = args.simfile

# Build default output filename from the input filename if not provided
if args.outputfile:
    outputfile = args.outputfile
else:
    base = os.path.basename(simfile)
    if base.endswith('.gz'):
        base = base[:-7]
    if base.endswith('.sim'):
        base = base[:-4]
    outputfile = f"{base}_MaterialInfo.txt"

# Setting up a dictionary to store the counts in each material
counts_in_material = {}

Reader = M.MFileEventsSim(Geometry)
if not Reader.Open(M.MString(simfile)):
    print(f"Failed to open {simfile}")
    sys.exit(1)

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

# Re-open the reader to iterate again and write out event to material mapping
if not Reader.Open(M.MString(simfile)):
    print(f"Failed to open {simfile}")
    sys.exit(1)

# Open file for writing
with open(outputfile, "w") as fout:
    fout.write("# EventID Material\n")

    while (Event := Reader.GetNextEvent()):
        M.SetOwnership(Event, True)
        EventID = Event.GetID()

        # Loop over interaction points in this event
        for i in range(Event.GetNIAs()):
            IA = Event.GetIAAt(i)
            if IA.GetProcess() != "PAIR":
                continue

            pos = IA.GetPosition()
            volume = Geometry.GetVolume(pos)
            if not volume:
                continue
            mat = volume.GetMaterial()
            if not mat:
                continue

            material_name = mat.GetName().ToString()

            # Write and stop after the first pair found
            fout.write(f"{EventID} {material_name}\n")
            break  
Reader.Close()

# Print results for each material in the terminal
print("---- Pair Conversions by Material ----")
total = sum(counts_in_material.values())
print(f"Total conversions: {total}")
for mat, count in sorted(counts_in_material.items(), key=lambda x: -x[1]):
    print(f"{mat}: {count}")

print(f"Done. Events saved to {outputfile}")

