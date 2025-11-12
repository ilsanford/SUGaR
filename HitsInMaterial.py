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

counter = 0
with open(outputfile, "w") as fout:
    fout.write("# EventID Material\n")

    while (Event := Reader.GetNextEvent()):
        M.SetOwnership(Event, True)
        for i in range(1,2):

            IA = Event.GetIAAt(i)
            counter += 1
            if counter % 10000 == 0:
                print(f"At event ID: {Event.GetID()}")

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
           
                    fout.write(f"{Event.GetID()} {material_name}\n")
        
Reader.Close()

# Print results for each material in the terminal
print("---- Pair Conversions by Material ----")
total = sum(counts_in_material.values())
print(f"Total conversions: {total}")
for mat, count in sorted(counts_in_material.items(), key=lambda x: -x[1]):
    print(f"{mat}: {count}")

print(f"Done. Events saved to {outputfile}")

# move the output writing into first loop
# make sure there is no double counting -> CsI -> first pair production - why is the modulation clean?