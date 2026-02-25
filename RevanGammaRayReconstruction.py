import numpy as np
import matplotlib.pyplot as plt
import argparse
import gzip

# function to normalize vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# computing the angle between reconstructed and actual
def angle_deg(v1, v2):
    cosang = np.dot(v1, v2)
    return np.degrees(np.arccos(cosang))

# function pulled from vertex finding script
def initial_vector_function(theta=0.0, phi=0.0):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    init_dir = np.array([np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad)])

    return normalize(init_dir)

# extracting info from revan output (tra file)
def extract_tra(filename, true_direction):
    angle_differences = []

    if filename.endswith(".gz"):
        f = gzip.open(filename, 'rt')
    else:
        f = open(filename, 'r')

    with f:
        Ee = Ep = None # energies
        ue = up = None # direction vectors

        for line in f:
            line = line.strip()
            if not line:  # skip blank lines
                continue

            if line.startswith("PE"): # pair event as identified by revan
                parts = line.split()
                if len(parts) < 6:  # must have at least 6 parts (some other tyes in tra file have a PE indicator)
                    continue
                try:
                    Ee = float(parts[1]) # energy
                    ue = np.array([float(parts[3]), float(parts[4]), float(parts[5])]) # vector
                except ValueError:
                    continue

            elif line.startswith("PP"):
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    Ep = float(parts[1])
                    up = np.array([float(parts[3]), float(parts[4]), float(parts[5])]) 
                except ValueError:
                    continue

            elif line.startswith("TQ"): # end of the event
                if Ee is not None and Ep is not None:
                    gamma = -(Ee * ue + Ep * up) / (Ee + Ep) # energy-weighted computation
                    gamma /= np.linalg.norm(gamma)
                    angle = angle_deg(gamma, true_direction)
                    angle_differences.append(angle)

                Ee = Ep = None
                ue = up = None

    return np.array(angle_differences)



def main():
    parser = argparse.ArgumentParser(description="Compute Revan reconstructed gamma-ray direction and plot histogram of angular differences.")

    parser.add_argument("inputfile", help="Input .tra file")
    parser.add_argument("--theta", type=float, default=0.0, help="MC theta in degrees (default=0)")
    parser.add_argument("--phi", type=float, default=0.0, help="MC phi in degrees (default=0)")

    args = parser.parse_args()

    true_direction = initial_vector_function(args.theta, args.phi)

    angles = extract_tra(args.inputfile, true_direction)

    # Naming convention for the output file based on the input file name
    if args.inputfile.endswith('.tra.gz'):
        base_filename = args.inputfile[:-7]
        filetype  = '.tra.gz'
    elif args.inputfile.endswith('.tra'):
        base_filename = args.inputfile[:-4]
        filetype = '.tra'
    else:
        raise ValueError('Input file must be of type .tra or .tra.gz')

    # Writing the angle differences to a text file for use in histogramming (see VertexIDAndPlottingMODIFIER.py)
    with open(f"{base_filename}_revan_angle_differences.txt", "w") as f:
        for angle in angles:
            f.write(f"{angle}\n")

    median = np.median(angles)
    if len(angles)  > 0:
        contain68 = np.percentile(angles, 68)
        contain95 = np.percentile(angles, 95)
    else:
        contain68 = None
        contain95 = None
        print("No reconstructed events found; cannot compute containment angles.")

    print("\n--------- OVERALL RECONSTRUCTION PERFORMANCE ---------")
    print(f"Total # of reconstructed events: {len(angles)}")
    print(f"Median angular difference: {median:.3f} deg")
    print(f"68% containment angle: {contain68:.3f} deg")
    #print(f"95% containment angle: {contain95:.3f} deg")
    print("----------------------------------------------\n")

    # histogram angles
    plt.figure()
    plt.hist(angles, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Angle difference between reconstructed and true gamma-ray direction [degrees]")
    plt.ylabel("Number of events")
    plt.title("Revan Gamma-Ray Reconstruction Accuracy")
    plt.grid(True)
    plt.show()
    # np.save("revan_angle_differences.npy", angles)


if __name__ == "__main__":
    main()

'''
Command line usage: python3 RevanGammaRayReconstruction.py <inputfile.tra.gz> --theta <MC theta in degrees> (default is 0) 
                                        --phi <MC phi in degrees> (default is 0)
'''