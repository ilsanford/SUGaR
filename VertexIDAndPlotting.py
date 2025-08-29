'''
Copyright (C) 2025  Isabella Sanford

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

A copy of the GNU Lesser General Public License is located in the
repository as LICENSE.md.
'''

'''
This file is separated into functions and classes:
    (a) Vertex class: defines properties of the vertex (position, ID, adjacent hits in the layer below, and the azimuthal angle)
    (b) Functions:
        i. Projects the initial gamma ray direction vector onto the next layer
        ii. Calculates and returns the distance from a hit to the projected/virtual point
        iii. Finds the two hits closest to the identified virtual hit
        iv. Plots the distribution of the distances from the hit to the virtual point
        v. Gets the hits in the layer below the identified vertex
    (b) VertexFinder class: locates the vertex - refer to the MERTrack.cxx file for the process of vertex identification
    (c) EventPlotting class: contains structure to plot the MC hits, RESE hits, MC vertex/interaction point, and identified vertex
                             for a specified event
    (d) MCInteraction class: structure to extract the MC interaction points for each event
'''

import ROOT as M
import gzip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import argparse
import numpy as np
from scipy.optimize import curve_fit
import os
import gc

'''
VERTEX CLASS
'''
class Vertex: 
    def __init__(self, rese, Geometry, AllRESEs):
        # Give RESE object to get the position information
        self.rese = rese
        self.Geometry = Geometry
        self.AllRESEs = AllRESEs

        pos = rese.GetPosition()
        self.x = pos.X()
        self.y = pos.Y()
        self.z = pos.Z()
        self.id = rese.GetID()

    # ------------------------------
    # Methods to access information:
    # ------------------------------

    def GetPosition(self):
        return np.array([self.x, self.y, self.z])

    def GetXPosition(self):
        return self.x

    def GetYPosition(self):
        return self.y

    def GetZPosition(self):
        return self.z

    def GetID(self):
        return self.id
    
    # ------------------------------
    # Compute azimuthal angle phi of the pair plane in photon's frame
    # hit1, hit2: hits after the vertex (e+/e- tracks)
    # theta, phi: polar/azimuthal angles of incoming photon in degrees
    # ref_direction: one of "RelativeX", "RelativeY", "RelativeZ" (defines reference axis)
    # ------------------------------

    def ComputePhi(self, theta, phi, hit1, hit2, ref_direction=None):

        # Define mapping from string to reference unit vector
        ref_map = {
            'RelativeX': np.array([1., 0., 0.]),
            'RelativeY': np.array([0., 1., 0.]),
            'RelativeZ': np.array([0., 0., 1.]),
        }

        # ------------------------------
        # Get positions of hits after the vertex
        # ------------------------------
        pos1 = np.array([
            hit1.GetPosition().X(),
            hit1.GetPosition().Y(),
            hit1.GetPosition().Z()
        ])
        pos2 = np.array([
            hit2.GetPosition().X(),
            hit2.GetPosition().Y(),
            hit2.GetPosition().Z()
        ])

        vertex_position = self.GetPosition()

        # Electron and positron direction vectors
        d1Dir = pos1 - vertex_position
        d2Dir = pos2 - vertex_position

        # Normalizing the direction vectors
        d1Dir /= np.linalg.norm(d1Dir)
        d2Dir /= np.linalg.norm(d2Dir)

        # ------------------------------
        # Construct initial photon direction from theta and phi
        # ------------------------------
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)

        initial_direction_full = np.array([x, y, z])
        init_dir = initial_direction_full / np.linalg.norm(initial_direction_full)

        # ------------------------------
        # Project reference axis into photon frame
        # ------------------------------
        refDir = ref_map[ref_direction]
        refDir_lab = refDir - np.dot(refDir, init_dir) * init_dir # Remove componenet along init_dir
        refDir_lab /= np.linalg.norm(refDir_lab)  # Normalize

        # Create second reference vector orthogonal to both
        ref2Dir = np.cross(init_dir, refDir_lab)
        ref2Dir /= np.linalg.norm(ref2Dir)

        # ------------------------------
        # Compute azimuthal angles of each track in rotated frame
        # ------------------------------
        phi1 = np.atan2(np.dot(ref2Dir, d1Dir), np.dot(refDir_lab, d1Dir)) + 2*np.pi
        phi2 = np.atan2(np.dot(ref2Dir, d2Dir), np.dot(refDir_lab, d2Dir)) + 2*np.pi

        phi = (phi1 + phi2) / 2
        if abs(phi1 - phi2) > np.pi:
            phi += np.pi
        
        # Wrap to [-pi, pi]
        return np.atan2(np.sin(phi), np.cos(phi))
          
'''
FUNCTIONS TO DETERMINE ELECTRON/POSITRON HITS
'''
def project_to_layer(init_pos, init_dir, z_target):

    # ------------------------------
    # Projects a virtual point along a direction vector to a plane at z = z_target.

    # Parameters:
    #    init_pos (np.ndarray): Initial position [x, y, z].
    #    init_dir (np.ndarray): Unit direction vector [dx, dy, dz].
    #    z_target (float): z-position of the detector layer.

    # Returns:
    #    np.ndarray: Projected point on the z = z_target plane.
    # ------------------------------

    t = (z_target - init_pos[2]) / init_dir[2] 
    projected_point = init_pos + t * init_dir
    return projected_point

def distance_to_virtual(position, virtual_point):

    # ------------------------------
    # Computes Euclidean distance from a hit to a virtual projected point.

    # Parameters:
    #    hit_pos (np.ndarray): Position of the hit [x, y, z].
    #    virtual_point (np.ndarray): Virtual projection point [x, y, z].

    # Returns:
    #    float: Distance in cm.
    # ------------------------------

    distance = np.linalg.norm(position - virtual_point)
    return distance

def select_two_closest_hits(hits, projected_point):

    # ------------------------------
    # Finds the two hits closest to the projected virtual point within a distance cutoff.

    # Parameters:
    #    hits (list): List of RESE hit objects.
    #    projected_point (np.ndarray): Virtual point from vertex projection.

    # Returns:
    #    tuple: (hit1, hit2, filtered_hits) where hit1 and hit2 are the closest hits (or None),
    #           and filtered_hits is a list of both (if found).
    # ------------------------------

    distance_cutoff = 5  # cm
    hits_with_dist = []

    for hit in hits:
        pos = np.array([
            hit.GetPosition().X(),
            hit.GetPosition().Y(),
            hit.GetPosition().Z()
        ])
        dist = distance_to_virtual(pos, projected_point)
        if dist <= distance_cutoff:
            hits_with_dist.append((hit, dist))

    # Require at least two hits within cutoff
    if len(hits_with_dist) < 2:
        return None, None, []

    # Sort by distance and select the two closest
    sorted_hits = sorted(hits_with_dist, key=lambda x: x[1])
    hit1, hit2 = sorted_hits[0][0], sorted_hits[1][0]
    filtered_hits = [hit1, hit2]

    return hit1, hit2, filtered_hits

def plot_distance_distribution(hits, projected_point, bins=50):

    # ------------------------------
    # Plots histogram of distances between each hit and the virtual point.

    # Parameters:
    #    hits (list of arrays): List of hit position arrays [x, y, z].
    #    projected_point (np.ndarray): Virtual projection point [x, y, z].
    #    bins (int): Number of bins in histogram.
    # ------------------------------

    distances = []
    for hit in hits:
        pos = np.array([
            hit[0],
            hit[1],
            hit[2]
        ])
        dist = distance_to_virtual(pos, projected_point)
        distances.append(dist)

    distances = np.array(distances)

    plt.figure(figsize=(8,5))
    plt.hist(distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Distance from virtual hit [cm]")
    plt.ylabel("Number of hits")
    plt.title("Distribution of distances of hits below vertex to virtual hit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_hits_in_layer_below(event_id, vertex_z, HTXPosition, HTYPosition, HTZPosition, layer_thickness=1.5):

    # ------------------------------
    # Identifies all hits in the detector layer directly below the vertex.

    # Parameters:
    #    event_id (int): Event ID to filter hits.
    #    vertex_z (float): z-position of the vertex.
    #    HTXPosition, HTYPosition, HTZPosition (list): HitTracker position data (in same units as vertex).
    #    layer_thickness (float): Expected vertical spacing between layers [cm].

    # Returns:
    #    list: List of 3D hit position arrays [x, y, z] in the layer below.
    # ------------------------------


    # Range of z values to look in (1 layer)
    target_z = vertex_z - layer_thickness
    tolerance = 0.1 # cm (to give some wiggle room)

    hits_in_layer = []

    xs = HTXPosition.get(event_id, [])
    ys = HTYPosition.get(event_id, [])
    zs = HTZPosition.get(event_id, [])

    for x, y, z in zip(xs, ys, zs):
        if abs(z - target_z) < tolerance:
            hits_in_layer.append(np.array([x, y, z]))

    return hits_in_layer

'''
THIS SECTION IS FOR FINDING VERTICES
'''
class VertexFinder:

    # ------------------------------
    # VertexFinder identifies potential gamma-ray conversion (vertex) points
    # in the detector using hit data from MEGAlib RESEs.
    # ------------------------------

    def __init__(self, Geometry, SearchRange = 30, NumberOfLayers = 2): # Let the default NumberOfLayers be 2

        # ------------------------------
        # Initialize VertexFinder.

        # Parameters:
        #    Geometry (MGeometry): Detector geometry for layer comparisons.
        #    SearchRange (int): Number of layers to search up/down from candidate vertex.
        #    NumberOfLayers (int): Minimum layers with 2+ hits required below vertex.
        # ------------------------------

        self.Geometry = Geometry
        self.SearchRange = SearchRange
        self.NumberOfLayers = NumberOfLayers
        self.DetectorList = [M.MDStrip2D] # AstroPix detectors only

    def IsInTracker(self, RESE):

        # ------------------------------
        # Determine if a RESE is part of the tracker.

        #  Returns:
        #    bool: True if RESE is in the tracker.
        # ------------------------------

        if RESE.GetType() == M.MRESE.c_Track:
            return True

        VS = RESE.GetVolumeSequence()
        Detector = VS.GetDetector()

        if Detector.__class__ in self.DetectorList: 
            return True

        return False

    def FindVertices(self, RE, theta, phi):
        # Note that a majority of this follows the logic implemented in the Revan code for vertex identification
        
        # ------------------------------
        # Main method to find vertex candidates in an event.

        # Parameters:
        #    RE (MRESEvent): The reconstructed event object.
        #    theta (float): Incoming photon polar angle (deg).
        #    phi (float): Incoming photon azimuthal angle (deg).

        # Returns:
        #    list: List of Vertex objects found in the event.
        # ------------------------------

        Vertices = []

        # Filter RESEs to those in tracker with energy > 0
        RESEs = [
            RE.GetRESEAt(i) for i in range(RE.GetNRESEs())
            if self.IsInTracker(RE.GetRESEAt(i)) and RE.GetRESEAt(i).GetEnergy() > 0
        ]

        # Sorting by depth in the tracker: shallowest -> deepest
        RESEs.sort(key=lambda rese: rese.GetPosition().Z())

        for i, candidate in enumerate(RESEs):
            # Determine if the hit is the only one in that layer
            OnlyHitInLayer = True
            for j in RESEs:
                if j == candidate:
                    continue
                if self.Geometry.AreInSameLayer(candidate, j):
                    OnlyHitInLayer = False
                    break
            if not OnlyHitInLayer:
                continue 

            NBelow = [0] * self.SearchRange
            NAbove = [0] * self.SearchRange

            for j in RESEs:
                if j == candidate:
                    continue
                Distance = self.Geometry.GetLayerDistance(candidate, j)
                if Distance > 0 and Distance < self.SearchRange:
                    NAbove[Distance] += 1
                if Distance < 0 and abs(Distance) < self.SearchRange:
                    NBelow[abs(Distance)] += 1
            
            # Looking for the vertex below ("inverted V")
            if NAbove[1] == 0:
                StartIndex = 0
                StopIndex = 0
                LayersWithAtLeastTwoHitsBetweenStartAndStop = 0

                for Distance in range(1, self.SearchRange-1):
                    if NBelow[Distance] == 0 and NBelow[Distance+1] == 0:
                        break
                    StopIndex = Distance

                    if StartIndex ==0 and NBelow[Distance] > 1 and NBelow[Distance+1] > 1:
                        StartIndex = Distance
                    
                    if StartIndex != 0:
                        if NBelow[Distance] >= 2:
                            LayersWithAtLeastTwoHitsBetweenStartAndStop += 1

                for Distance in range(StopIndex, 2, -1):
                    if NBelow[Distance-1]>= 2 and NBelow[Distance-2] >= 2:
                        break
                    StopIndex = Distance

                interlayerdistance = 1.5 #cm (change depending on detector geometry, this is for AMEGO-X)

                # Searching through the number of layers after the layer in which the vertex is identified
                if LayersWithAtLeastTwoHitsBetweenStartAndStop >= self.NumberOfLayers:

                    # Project along MC direction to the next layer
                    init_pos = np.array([
                        candidate.GetPosition().X(),
                        candidate.GetPosition().Y(),
                        candidate.GetPosition().Z()
                    ])

                    theta_rad = np.deg2rad(theta)
                    phi_rad = np.deg2rad(phi)

                    x = np.sin(theta_rad) * np.cos(phi_rad)
                    y = np.sin(theta_rad) * np.sin(phi_rad)
                    z = np.cos(theta_rad)

                    init_dir = np.array([x,y,z])/np.linalg.norm(np.array([x, y, z]))

                    # Move to the layer below candidate vertex and project virtual point on that layer
                    z_target = candidate.GetPosition().Z() - interlayerdistance
                    projected_point = project_to_layer(init_pos, init_dir, z_target)

                    # Get all hits in z_target layer
                    hits_below = [rese for rese in RESEs if self.Geometry.GetLayerDistance(candidate, rese) == -1]

                    # Apply distance selection
                    hit1, hit2, filtered_hits = select_two_closest_hits(hits_below, projected_point)
                    if hit1 is None or hit2 is None:
                        continue

                    if hit1 and hit2:
                        # Construct vertex with only the selected hits
                        vtx = Vertex(candidate, self.Geometry, [hit1, hit2])
                        Vertices.append(vtx)
                                    
                    else:
                        pass
                    
        return Vertices

    def PlotVertexHistogram(self, NumberOfVerticesPerEvent, inputfile):

        # ------------------------------
        # Plot histogram of the number of vertices per event.

        # Parameters:
        #    NumberOfVerticesPerEvent (list): List of vertex counts per event.
        #    inputfile (str): Filename prefix for saving the plot.
        # ------------------------------

        if not NumberOfVerticesPerEvent:
            print("No vertex data to plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(NumberOfVerticesPerEvent, bins=np.arange(0, max(NumberOfVerticesPerEvent)+2), edgecolor='black')
        plt.xlabel("Number of vertices per event")
        plt.ylabel("Number of events")
        plt.title(f"Histogram: Number of Vertices per Gamma-Ray Event - {self.NumberOfLayers} Layers")
        plt.tight_layout()
        plt.savefig(f"{inputfile}_VerticesHistogram_{self.NumberOfLayers}Layers.png", dpi=300)
        plt.show()

    def TopVertex(self, vertex_list):

        # ------------------------------
        # Select the top-most vertex (max Z).

        # Parameters:
        #    vertex_list (list): List of Vertex objects.

        # Returns:
        #    Vertex: The vertex with the highest Z position.
        # ------------------------------

        return max(vertex_list, key=lambda v: v.GetZPosition())        

'''
THIS SECTION IS FOR EVENT PLOTTING
'''
class EventPlotting:

    # ------------------------------
    # Handles plotting and data extraction from simulated and reconstructed gamma-ray events.
    # Used for comparing true (simulated) and reconstructed hits and vertices.
    # ------------------------------

    def __init__(self, inputfile, Geometry):
        self.inputfile = inputfile
        self.Geometry = Geometry

    def GetSimHits(self):

        # ------------------------------
        # Extract simulated HT hits from a .sim.gz file.

        # Returns:
        #    tuple of dicts: (HTXPosition, HTYPosition, HTZPosition) per event ID.
        # ------------------------------

        HTXPosition = {}
        HTYPosition = {}
        HTZPosition = {}
        CurrentEventID = None
        ExpectEventID = False

        with gzip.open(self.inputfile, 'rt') as f:
            for line in f:
                line = line.strip()

                # New event starts
                if line.startswith('SE'):
                    ExpectEventID = True
                    continue

                if ExpectEventID:
                    if line.startswith('ID'):
                        HTColumns = line.split()
                        if len(HTColumns) >= 2:
                            try:
                                CurrentEventID = int(HTColumns[1])
                                HTXPosition[CurrentEventID] = []
                                HTYPosition[CurrentEventID] = []
                                HTZPosition[CurrentEventID] = []
                            except ValueError:
                                CurrentEventID = None
                    ExpectEventID = False

                # HT line
                if line.startswith('HTsim 1'):
                    HTColumns = line.split(';')

                    try:
                            XPositionColumn = float(HTColumns[1])
                            YPositionColumn = float(HTColumns[2])
                            ZPositionColumn = float(HTColumns[3])
                            if CurrentEventID is not None:
                                HTXPosition[CurrentEventID].append(XPositionColumn)
                                HTYPosition[CurrentEventID].append(YPositionColumn)
                                HTZPosition[CurrentEventID].append(ZPositionColumn)
                    except (IndexError, ValueError):
                        continue

        return HTXPosition, HTYPosition, HTZPosition

    def GetRESEEvents(self, Geometry, inputfile, MaxNumberOfEvents=None):

        # ------------------------------
        # Extract clustered and noised RESE hits from a SIM file using MEGAlib evta reader

        # Parameters:
        #    Geometry (MGeometry): Detector geometry object.
        #    inputfile (str): Path to the SIM file.
        #    MaxNumberOfEvents (int, optional): Stop after this many events.

        # Returns:
        #    dict: {EventID: [(X, Y, Z, marker, label)]}
        # ------------------------------

        # Getting the noised and clustered events
        Reader = M.MFileEventsEvta(self.Geometry)
        Reader.Open(M.MString(self.inputfile))

        Clusterizer = M.MERHitClusterizer()
        Clusterizer.SetParameters(1, -1) # Default clustering parameters

        RESEData = {} 
        EventCount = 0

        while True:
            Event = Reader.GetNextEvent()
            if not Event or not Event.IsValid():
                break
            if MaxNumberOfEvents and EventCount >= MaxNumberOfEvents:
                break

            REIEvent = M.MRawEventIncarnations()
            REIEvent.SetInitialRawEvent(Event)
            EventID = Event.GetEventID()
            Clusterizer.Analyze(REIEvent)
            EventAfterClustering = REIEvent.GetInitialRawEvent()

            # Assign data to respective event ID number
            RESEData[EventID] = []

            NumberOfRESEs = EventAfterClustering.GetNRESEs()

            for i in range(NumberOfRESEs):
                RESEEvent = EventAfterClustering.GetRESEAt(i)

                # Skip unphysical zero-energy hits
                if RESEEvent.GetEnergy() <= 0:
                    continue

                Position = RESEEvent.GetPosition()
                X, Y, Z = Position.X(), Position.Y(), Position.Z()

                RESEClass = RESEEvent.IsA()
                if RESEClass.InheritsFrom(M.MREHit.Class()):
                    marker = 'o'
                    label = "Hit"
                elif RESEClass.InheritsFrom(M.MRECluster.Class()):
                    marker = 's'
                    label = "Cluster"
                elif RESEClass.InheritsFrom(M.MRETrack.Class()):
                    marker = '^'
                    label = "Track"
                else:
                    marker = 'x'
                    label = "Unknown"
                 

                RESEData[EventID].append((X, Y, Z, marker, label))

            EventCount += 1

        Reader.Close()
        return RESEData

    def PlottingSimAndRESEs(self, EventID, HTX, HTY, HTZ, RESEHits, Vertices, MCPoints=None):

        # ------------------------------
        # Plot 3D view of simulated (MC) hits, RESE hits, identified vertices, and true MC vertex.

        # Parameters:
        #    EventID (int): ID of the event to visualize.
        #    HTX, HTY, HTZ (dict): Simulated hit positions.
        #    RESEHits (dict): Clustered RESE hits from reconstruction.
        #    Vertices (list): Vertex objects for this event.
        #    MCPoints (dict, optional): True MC interaction point per event.
        # ------------------------------

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Event ID {EventID} : MC Hits and RESEs")
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        ax.set_zlabel("Z [cm]")

        # Plot simulated tracker hits
        if EventID in HTX and HTX[EventID]:
            ax.scatter(HTX[EventID], HTY[EventID], HTZ[EventID], c='blue', marker='o', label='Simulated HT Events', alpha=0.4, s=40)

        # Plot all RESE events in red with different markers by RESE type
        if EventID in RESEHits and RESEHits[EventID]:
            for x, y, z, marker, label in RESEHits[EventID]:
                ax.scatter(x, y, z, c='red', marker=marker, alpha=0.4, s=40)

        # Plotting the identified recontructed vertices
        if Vertices:
            VertexXPos = [v.GetXPosition() for v in Vertices]
            VertexYPos = [v.GetYPosition() for v in Vertices]
            VertexZPos = [v.GetZPosition() for v in Vertices]

            ax.scatter(VertexXPos, VertexYPos, VertexZPos, color='black', marker='x', s=200, label='Identified Vertex', zorder=10)
        
        # Plotting the MC interaction point/vertex (the "true" vertex)
        if MCPoints is not None and EventID in MCPoints:
            mc_pos = MCPoints[EventID]
            ax.scatter(mc_pos[0], mc_pos[1], mc_pos[2], color='green', marker='x', s=200, label='MC Interaction Point', zorder=11)

        # Creating the different labels in the legend
        RedClusterLegend = Line2D([0], [0], marker='s', color='red', linestyle='None', markersize=8, label='Clustered RESEs')
        RedHTLegend = Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=8, label='Hit RESEs')
        BlueLegend = Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=8, label='MC Hits')
        VertexLegend = Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Identified Vertex Position(s)')
        MCPointLegend = Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=10, label='MC Vertex')

        # Combine into final legend list depending on what's present
        legend_handles = [BlueLegend]
        if EventID in RESEHits and RESEHits[EventID]:
            legend_handles += [RedHTLegend, RedClusterLegend]
        if Vertices:
            legend_handles += [VertexLegend]
        if MCPoints is not None and EventID in MCPoints:
            legend_handles += [MCPointLegend]
        

        ax.legend(handles=legend_handles, loc='upper right')
    
        plt.tight_layout()
        plt.show()

'''
ALSO WANT TO EXTRACT "TRUE" INTERACTION POINT
'''
class MCInteraction: # does not NEED to be a class -> this is mostly just for organizational purposes

    def GetMCInteractionPoints(inputfile):

        # ------------------------------
        # Extracts the 'true' interaction points from a .sim.gz file.

        # Args:
        #    inputfile (str): Path to the input .sim.gz file.

        # Returns:
        #    dict: Dictionary mapping EventID to its true MC interaction point as a NumPy array [x, y, z].
        # ------------------------------

        MCPoints = {}
        CurrentEventID = None
        ExpectEventID = False
        FoundIA = False

        with gzip.open(inputfile, 'rt') as f:
            for line in f:
                line = line.strip()
                if line.startswith('SE'):
                    ExpectEventID = True
                    FoundIA = False
                    continue

                if ExpectEventID:
                    if line.startswith('ID'):
                        columns = line.split()
                        if len(columns) >= 2:
                            try:
                                CurrentEventID = int(columns[1])
                            except ValueError:
                                CurrentEventID = None
                    ExpectEventID = False
                    continue

                # Look for first IA line of type PAIR
                if not FoundIA and line.startswith('IA PAIR'):
                    IAColumns = line.split(';')
                    try:
                        InteractionPositionX = float(IAColumns[4])
                        InteractionPositionY = float(IAColumns[5])
                        InteractionPositionZ = float(IAColumns[6])
                        if CurrentEventID is not None:
                            MCPoints[CurrentEventID] = np.array([InteractionPositionX, InteractionPositionY, InteractionPositionZ])
                            FoundIA = True
                    except (IndexError, ValueError):
                        continue

        return MCPoints

    def PlotVertexResiduals(MCPoints, VertexDict, EventNumberToEventID=None):

        # ------------------------------
        # Plots residuals between identified vertex positions and true MC interaction points.

        # Args:
        #    MCPoints (dict): Dictionary of true MC points from `get_mc_interaction_points`.
        #    VertexDict (dict): Dictionary mapping EventID to list of identified Vertex objects.
        #    EventNumberToEventID (dict, optional): Mapping the event number to the event ID.
        # ------------------------------

        Residuals = []
        EventResidualPairs = []
        dx_list, dy_list, dz_list = [], [], []

        for event_id, MCPosition in MCPoints.items():
            if event_id in VertexDict and VertexDict[event_id]:
                first_vertex = VertexDict[event_id][0]
                VPos = first_vertex.GetPosition()
                dx, dy, dz = VPos - MCPosition
                d = np.linalg.norm([dx, dy, dz])
                Residuals.append(d)
                EventResidualPairs.append((event_id, d))
                dx_list.append(dx)
                dy_list.append(dy)
                dz_list.append(dz)


        if not Residuals:
            print("No Residuals to plot.")
            return

        # Print the event with the largest residual
        max_event_id, max_distance = max(EventResidualPairs, key=lambda x: x[1])
        if EventNumberToEventID is not None:
            matching_event_numbers = [num for num, eid in EventNumberToEventID.items() if eid == max_event_id]
            if matching_event_numbers:
                print(f"Event with largest residual: Event Number {matching_event_numbers[0]}, Event ID {max_event_id}, distance = {max_distance:.2f} cm")
            else:
                print(f"Event with largest residual: Event ID {max_event_id}, distance = {max_distance:.2f} cm (event number unknown)")
        else:
            print(f"Event with largest residual: Event ID {max_event_id}, distance = {max_distance:.2f} cm")


        # Plotting
        plt.figure(figsize=(8, 5))
        plt.hist(Residuals, bins=np.arange(0, 100, 0.05), edgecolor='black')
        plt.xlabel("Distance between MC Interaction Point and Identified First Vertex [cm]")
        plt.ylabel("Number of Events")

        plt.title("Distribution of Differences Between MC Vertex and Identified Vertex Position")
        plt.tight_layout()
        plt.show()

        # x, y, z histograms
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.hist(dx_list, bins=np.arange(-20, 20, 0.05), edgecolor='black')
        plt.xlabel(r"$\Delta$ x [cm]")
        plt.title("Residual in X")

        plt.subplot(1, 3, 2)
        plt.hist(dy_list, bins=np.arange(-20, 20, 0.05), edgecolor='black')
        plt.xlabel(r"$\Delta$ y [cm]")
        plt.title("Residual in Y")

        plt.subplot(1, 3, 3)
        plt.hist(dz_list, bins=np.arange(-50, 50, 0.05), edgecolor='black')
        plt.xlabel(r"$\Delta$ z [cm]")
        plt.title("Residual in Z")

        plt.suptitle('Residuals in X, Y, and Z')
        plt.tight_layout()
        plt.show()

        # Scatterplot in x and y colored by dz
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(dx_list, dy_list, c=dz_list, cmap='coolwarm', alpha=0.5, edgecolor='k', s=50)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel(r"$\Delta$ x [cm]")
        plt.ylabel(r"$\Delta$ y [cm]")
        plt.title(r"XY Vertex Residuals Colored by $\Delta$ z")
        plt.colorbar(sc, label=r"$\Delta$ z [cm]")
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

'''
FUNCTION FOR THE MODULATION FIT -> need? have sep. file
'''
def polarfit(x, A, phi0, N):

    # ------------------------------
    # Fit function for azimuthal modulation of pair production events.
    
    # Parameters:
    #    phi (float or ndarray): Azimuthal angle(s) in radians.
    #    A (float): Modulation amplitude.
    #    phi0 (float): Modulation phase offset (in radians).
    #    N (float): Normalization factor

    # Returns:
    #    float or ndarray: Value(s) of the modulation function at phi.
    # ------------------------------

    return N / (2 * np.pi) * (1 - A * np.cos(2 * (x - phi0)))

'''
MAIN FUNCTION FOR EXECUTION IN TERMINAL
'''
if __name__ == "__main__":

    # Load in and initialize MEGAlib 
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    G = M.MGlobal()
    G.Initialize()

    # ------------------------------
    # Command line argument options with descriptions for each.
    # ------------------------------

    parser = argparse.ArgumentParser(description="Process a .sim.gz file from MEGAlib Cosima output.")
    parser.add_argument("inputfile", help="Enter path to desired input file.")
    parser.add_argument("--plot-event-number", type=int, default=None, help="Enter the desired event number (not event ID) to be plotted.")
    parser.add_argument("--plot-eventID", type=int, default = None, help="Enter the desired event ID (not event number) to be plotted.  ")
    parser.add_argument('--layers', type=int, default=2, help='Enter desired number of layers to classify vertex. Default is 2.')
    parser.add_argument('--plot-histogram', action='store_true', help='Plot number of vertices per event histogram.')
    parser.add_argument('--plot-residuals', action='store_true', help='Plot residuals between MC and reconstructed vertex.')
    parser.add_argument('--plot-events', action='store_true', help='Plot individual event visualizations.')
    parser.add_argument('--ref-dir', type=str, default='RelativeX', choices=['RelativeX', 'RelativeY', 'RelativeZ'], help="Reference direction used for azimuthal angle calculation (default: RelativeX)")
    parser.add_argument('--theta', type=float, default=0.0, help='Polar angle theta (in degrees) of the incoming gamma-ray (default: 0 degrees)')
    parser.add_argument('--phi', type=float, default=0.0, help='Azimuthal angle phi (in degrees) of the incoming gamma-ray (default: 0 degrees)')
    parser.add_argument('--plot-distance-dist', action='store_true', help='Plot distance distribution of hits relative to virtual hit in the layer below vertex')
    
    # Loading in the required geometry (AMEGO-X for this analysis)
    GeometryName = "../../MEGAlib_Data/Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
    Geometry = M.MGeometryRevan()
    if Geometry.ScanSetupFile(M.MString(GeometryName)):
        print("Geometry " + GeometryName + " loaded!")
    else:
        print("Unable to load geometry " + GeometryName + " - Aborting!")
        quit()


    args = parser.parse_args()

    inputfile = args.inputfile
    plot_event_number = args.plot_event_number
    NumberOfLayers = args.layers

    # Read in events (note that MFileEventsEvta automatically applies noising)
    Reader = M.MFileEventsEvta(Geometry)
    Reader.Open(M.MString(inputfile))

    # Identify vertices in the input data
    VF = VertexFinder(Geometry, NumberOfLayers=NumberOfLayers)

    # Cluster the events
    Clusterizer = M.MERHitClusterizer()
    Clusterizer.SetGeometry(Geometry)
    Clusterizer.SetParameters(1, -1)
    Clusterizer.PreAnalysis()    

    # Initialize lists and dicts
    EventCount = 0
    PairsFound = 0
    NumberOfVerticesPerEvent = []
    EventNumberToEventID = {}
    VertexDict = {}
    event_id_list = []  
    phi_values = []
    all_distances_to_virtual = []
    passing_distances = []

    # Setting up the structure to write out events
    block_size = 10000 # Write out events in blocks of 10,000
    output_phi_filename = f"{inputfile}_PhiValues.txt"

    # Remove output file if it already exists
    if os.path.exists(output_phi_filename):
        os.remove(output_phi_filename)

    # Reading each event and stopping if none left, rejecting "invalid" events (as identified in MEGAlib)
    while True:
        RE = Reader.GetNextEvent()
        M.SetOwnership(RE, True) # necessary to avoid memory leaks

        if not RE:
            print("No more events.")
            break
        if not RE.IsValid():
            print(f"Skipping invalid event at count {EventCount}.") # this seems to correspond to an energy deposit of 0 keV
            continue

        EventNumberToEventID[EventCount] = RE.GetEventID()
        REI = M.MRawEventIncarnations()
        M.SetOwnership(REI, True)

        REI.SetInitialRawEvent(RE)
        Clusterizer.Analyze(REI)
        ClusteredRE = REI.GetInitialRawEvent()
        M.SetOwnership(ClusteredRE, True)

        # Getting the vertices and assigning each to its corresponding event ID -> VertexDict
        Vertices = VF.FindVertices(ClusteredRE, theta=args.theta, phi=args.phi)
        VertexDict[RE.GetEventID()] = Vertices 

        if Vertices and args.plot_distance_dist: # CHANGE
            EP = EventPlotting(inputfile, Geometry=Geometry)
            HTXPosition, HTYPosition, HTZPosition = EP.GetSimHits()

            top_vertex = VF.TopVertex(Vertices)
            vertex_z = top_vertex.GetPosition()[2]

            hits_below = get_hits_in_layer_below(RE.GetEventID(), vertex_z, HTXPosition, HTYPosition, HTZPosition)

            # Projected point -> make function
            theta_rad = np.radians(args.theta)
            phi_rad = np.radians(args.phi)
            init_dir = np.array([
                np.sin(theta_rad) * np.cos(phi_rad),
                np.sin(theta_rad) * np.sin(phi_rad),
                np.cos(theta_rad)
            ])
            init_pos = np.array([
                top_vertex.GetPosition()[0],
                top_vertex.GetPosition()[1],
                vertex_z
            ])

            z_target = vertex_z - 1.5 # cm (interlayer distance)
            projected_point = project_to_layer(init_pos, init_dir, z_target)

            # Plot distribution 
            for hit in hits_below:
                distance = distance_to_virtual(hit, projected_point)
                all_distances_to_virtual.append(distance)

                if distance < 5.0:
                    passing_distances.append(distance)
            
            phi = top_vertex.ComputePhi(theta = args.theta, phi = args.phi, hit1=top_vertex.AllRESEs[0], hit2=top_vertex.AllRESEs[1], ref_direction = args.ref_dir)
            #if phi is not None:
            event_id_list.append(RE.GetEventID())
            phi_values.append(phi)

        else:
            pass
        

        NumberOfVerticesPerEvent.append(len(Vertices))

        if len(Vertices) > 0:
            PairsFound += 1
        
        EventCount += 1

        # if the event count is divisible by block_size (ex. 10,000) then print all of that to a text file -> repeat until all events are written
        if EventCount % block_size == 0:
            with open(output_phi_filename, 'a') as f:
                for eid, phi in zip(event_id_list, phi_values):
                    f.write(f"{eid} {phi}\n")
            phi_values.clear() # can remove if wanted
            event_id_list.clear()
            gc.collect()
            print(f"Processed {EventCount} events...", flush=True)

    # Write out the leftover azimuthal angles into the output file
    if event_id_list:
        with open(output_phi_filename, 'a') as f:
            for eid, phi in zip(event_id_list, phi_values):
                f.write(f"{eid} {phi}\n")

    if args.plot_distance_dist and all_distances_to_virtual:

        plt.hist(passing_distances, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Distance to virtual point (cm)')
        plt.ylabel('Counts')
        plt.title('Distribution of hit distances (passing < 5 cm)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(all_distances_to_virtual, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("Distance from virtual hit [cm]")
        plt.ylabel("Number of hits")
        plt.title("Combined Distance Distribution to Virtual Hit")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"Scanned {EventCount} events")

    if phi_values:
        plt.figure(figsize=(8, 5))
        plt.hist(phi_values, bins=np.linspace(-np.pi, np.pi, 17), range=(-np.pi, np.pi), edgecolor='black', density=True)
        plt.xlabel(r"Azimuthal angle $\phi$")
        plt.ylabel('Counts')
        plt.title(r"Distribution of Azimuthal Angle $\phi$")
        plt.tight_layout()
        plt.show()

        hist, bins = np.histogram(phi_values, bins=np.linspace(-np.pi, np.pi, 17))
        yerr = np.sqrt(hist)
        x = .5*(bins[:-1]+bins[1:]) # bin centers
        plt.errorbar(x, hist, yerr=yerr, label="data", fmt ='o')

        try:
            popt, pcov = curve_fit(polarfit, x, hist, sigma=yerr, absolute_sigma=True)
            A_fit, phi0_fit, N_fit = popt
            dA, dphi0, dN = np.sqrt(np.diag(pcov))
            print(f"A     = {A_fit:.3f} ± {dA:.3f}")
            print(f"phi_0 = {phi0_fit:.3f} ± {dphi0:.3f} rad")
            print(f"N     = {N_fit:.1f} ± {dN:.1f}")

            # Plot the fit
            xx = np.linspace(-np.pi, np.pi, 500)
            plt.plot(xx, polarfit(xx, *popt), 'r--', label='Polarization Fit')

        except RuntimeError:
            print("Fit failed. Not enough statistics or poor modulation.")

        plt.legend()
        plt.xlabel(r"Azimuthal angle $\phi$ [rad]")
        plt.ylabel('Counts per bin')
        plt.title('Polarization Modulation Fit')
        plt.tight_layout()
        plt.show()
        print('Number of phi values:', len(phi_values))


    if args.plot_histogram:
        VF.PlotVertexHistogram(NumberOfVerticesPerEvent, inputfile)

    if args.plot_events:

        MCPoints = MCInteraction.GetMCInteractionPoints(inputfile)
            
        '''
        NOW THE EVENT PLOTTING FOR CHOSEN EVENT
        '''
        EP = EventPlotting(inputfile, Geometry=Geometry)
        # Go through the sim hits
        HTX, HTY, HTZ = EP.GetSimHits()
        print(f"Parsed sim hits for {len(HTX)} events")

        # Go through RESEs
        RESEHits = EP.GetRESEEvents(Geometry, inputfile, MaxNumberOfEvents=None) # set some max number of events if desired
        print(f"Parsed RESEs for {len(RESEHits)} events")

        # Plotting both on same plot for corresponding event ID
        SharedEventID = sorted(set(HTX.keys()).intersection(RESEHits.keys()))
        print(f"Plotting {len(SharedEventID)} events with both simulated and clustered data")
                
        if args.plot_eventID is not None:
            event_id_to_plot = args.eventID
            print(f"Plotting event with Event ID {event_id_to_plot}")
            EP.PlottingSimAndRESEs(event_id_to_plot, HTX, HTY, HTZ, RESEHits, VertexDict.get(event_id_to_plot, []), MCPoints=MCPoints)

        elif plot_event_number is not None:
            if plot_event_number in EventNumberToEventID:
                event_id_to_plot = EventNumberToEventID[plot_event_number]
                print(f"Plotting event number {plot_event_number} (Event ID {event_id_to_plot})")
                EP.PlottingSimAndRESEs(event_id_to_plot, HTX, HTY, HTZ, RESEHits, VertexDict.get(event_id_to_plot, []), MCPoints=MCPoints)
            else:
                print(f"Requested event number {plot_event_number} is out of bounds.")
        else:
            print("No specific event specified. Plotting all events.")
            for Event_ID in SharedEventID:
                EP.PlottingSimAndRESEs(Event_ID, HTX, HTY, HTZ, RESEHits, VertexDict.get(Event_ID, []), MCPoints=MCPoints)
    
    if args.plot_residuals:
        MCPoints = MCInteraction.GetMCInteractionPoints(inputfile)
        MCInteraction.PlotVertexResiduals(MCPoints, VertexDict, EventNumberToEventID)