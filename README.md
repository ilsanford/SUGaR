# SUGaR-PR: Software to Update Gamma-ray Reconstruction in the Pair Regime
### A Python MEGAlib extension for improving gamma-ray reconstruction in the pair regime as well as performing pair regime polarimetry
Python scripts designed to build off of existing MEGAlib software to perform analysis on gamma rays undergoing pair production in a detector. This performs vertex identification, gamma-ray direction reconstruction, and azimuthal angle computation (for polarimetry). Currently, this is specifically applied to the AMEGO-X geometry but parameters can be changed to adapt to different geometries.

## File Locations
Any simulation files that being used should be in the same directory as the files in this repository. For the SUGaR files to load the proper geometry, the AMEGO-X geometry setup file should be located here (starting from the directory containing the SUGaR files):

``` .../Geometry/AMEGO_Midex/AmegoXBase.geo.setup ```

## Objectives
This software has the ability to do the following:
  1. Filter out all true (MC) pair events in a simulation (SIM) file into a file containing only pair events.
  2. Reconstruct vertex locations and gamma ray directions for pair events that satisfy a certain set of criteria (see code documentation for a more detailed description).
  3. Determine which material a given event converted in. Vertex finding and gamma-ray direction reconstruction can then be applied by material.
  4. Determine if a conversion occured in dead/passive material. Analysis can then be run on just these events as well.

## The Pipeline
Depending on the desired analysis, there are multiple ways to run the files in this software. Below is a flowchart that shows the options.


<img width="3316" height="2472" alt="SUGaR Flowchart drawio (1)" src="https://github.com/user-attachments/assets/997007d4-0165-45fc-b37c-3b42a1b6941e" />



