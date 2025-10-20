# SUGaR-PRP
## Software for Understanding Gamma-Ray Pair Regime Polarimetry
### A Python MEGAlib extension for gamma-ray pair regime polarimetry
Python scripts designed to build off of existing MEGAlib software to perform analysis on polarized gamma rays undergoing pair production in a detector. Currently, this is specifically applied to the AMEGO-X geometry but parameters can be changed to adapt to different geometries.

### The Pipeline
1. Selecting the Monte Carlo pair events: *PairEventSelection.py* OR *PairEventSelection_MultipleFiles.py* -> If parallel runs are done (using mcosima), use the latter file for event filtering. If not, use the first.
2. Identifying vertices, computing azimuthal angles, and plotting: *VertexIDAndPlotting.py* -> Takes the file containing only pair events and computes the azimuthal angle for all identified events. The script also automatically applies detector effects and a clustering algorithm (both are implemented in MEGAlib). See the file to know additional plotting options.
3. 

