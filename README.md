# SUGaR-PRP
## Software to Update Gamma-ray Reconstruction in the Pair Regime
### A Python MEGAlib extension for gamma-ray pair regime polarimetry
Python scripts designed to build off of existing MEGAlib software to perform analysis on gamma rays undergoing pair production in a detector. Currently, this is specifically applied to the AMEGO-X geometry but parameters can be changed to adapt to different geometries.

### The Pipeline
1. Selecting the Monte Carlo pair events: *PairEventSelection.py* OR *PairEventSelection_MultipleFiles.py* -> If parallel runs are done (using mcosima), use the latter file for event filtering. If not, use the first.
2. Identifying vertices, computing azimuthal angles, and plotting: *VertexIDAndPlotting.py* -> Takes the file containing only pair events and computes the azimuthal angle for all identified events. The script also automatically applies detector effects and a clustering algorithm (both are implemented in MEGAlib). See the file to know additional plotting options.
3. Determining the modulation: *ModulationFit.py* -> Takes the output text file containing azimuthal angle values from the previous *VertexIDAndPlotting.py* script. Requires both a polarized and an unpolarized file for use. Plots a histogram of azimuthal angles for both and then computes their ratio, plots a histogram of that ratio, fits a sine curve to the data and extracts the fit parameters to determine the modulation.

## Additional Options
If interested in the number of hits in a certain material as well as the modulation in that material, see the *HitsInMaterial.py* file. This file takes an input SIM file containing only pair events, extracts the locations of the pair conversions, and references that location with the corresponding material in that location from the geometry file to determine in which material the pair conversion occurred. Running this script will output a list of materials with the number of conversion in each and will also save a corresponding text file that contains event ID and associated material for that event's conversion.

If looking to understand the modulation in a material, the azimuthal angles for all of the event IDs in the *HitsInMaterial.py* file are needed. The *MaterialVertexAnalysis.py* files does this by taking the output text file of event ID and material, filtering out event IDs only associated with the chosen material, referencing those event IDs to the original SIM file, and then performing azimuthal angle calculations on those events (using the process implemented in the *VertexIDAndPlotting.py* script). The output from this is a text file containing a list of phi values. This text file can then be passed into the *ModulationFit.py* script to get histograms and modulation values for pair conversions in a certain material.

