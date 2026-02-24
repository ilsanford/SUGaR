'''
This file is separated into standalone functions and classes:
    (a) Vertex class: defines properties of the vertex (position, ID, adjacent hits in the layer below, associated
                                                        gamma ray direction, and the azimuthal angle)
    (b) Functions:
        i. Uses input theta and phi (MC info) to compute the initial gamma ray direction vector
        ii. Projects the initial gamma ray direction vector onto the next layer
        iii. Calculates and returns the distance from a hit to the projected/virtual point
        iv. Finds the two hits closest to the identified virtual hit
        v. Plots the distribution of the distances from the hit to the virtual point
        vi. Gets the hits in the layer below the identified vertex
    (c) VertexFinder class: locates the vertex - refer to the MERTrack.cxx file for the process of vertex identification. Also
                            reconstructs the gamma-ray direction using the identified vertex and hits in the layer below. 
    (d) EventPlotting class: contains structure to plot the MC hits, RESE hits, MC vertex/interaction point, and identified vertex
                            for a specified event
    (e) MCInteraction class: structure to extract the MC interaction points for each event
'''

'''
IMPORTING NECESSARY LIBRARIES AND MODULES
'''
import ROOT as M
import gzip
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm 
from matplotlib.lines import Line2D
import argparse
import numpy as np
import os
import gc

'''
(a) VERTEX CLASS
'''
class Vertex: 
    def __init__(self, rese, Geometry, AllRESEs, position=None):
        # Give RESE object to get the position information
        self.rese = rese
        self.Geometry = Geometry
        self.AllRESEs = AllRESEs

        if position is not None:
            self.x, self.y, self.z = position
            self.id = rese.GetID() if rese else -1 # just assign an event ID since this is not an RESE (make more robust at some point?)

        else:
            pos = rese.GetPosition()
            self.x = pos.X()
            self.y = pos.Y()
            self.z = pos.Z()
            self.id = rese.GetID()

        self.energy = rese.GetEnergy() if rese is not None else None

        # giving track reconstruction placeholders for now
        self.electron_dir = None
        self.positron_dir = None
        self.electron_energy = None
        self.positron_energy = None
        self.gamma_dir = None

    # Methods to access information:
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
    
    def GetHitEnergy(self):
        return self.energy
    
    def ComputePhi(self, theta, phi, hit1, hit2, ref_direction=None):
        '''
        Compute azimuthal angle (phi) of the pair plane in photon's frame
        ------------------------------------------------------------------------------
        Parameters:
            hit1, hit2: hits after the vertex (electron/positron tracks)
            theta, phi: polar/azimuthal angles of incoming photon in degrees
            ref_direction: one of "RelativeX", "RelativeY", "RelativeZ" inputs (defines reference axis)

        Returns:
            phi: azimuthal angle of the pair plane in photon's frame (radians, wrapped to [-pi, pi])
        '''

        # Define mapping from input string to reference unit vector
        ref_map = {'RelativeX': np.array([1., 0., 0.]),
            'RelativeY': np.array([0., 1., 0.]),
            'RelativeZ': np.array([0., 0., 1.]),}

        # Get positions of hits after the vertex
        pos1 = np.array([hit1.GetPosition().X(),
            hit1.GetPosition().Y(),
            hit1.GetPosition().Z()])
        pos2 = np.array([hit2.GetPosition().X(),
            hit2.GetPosition().Y(),
            hit2.GetPosition().Z()])

        vertex_position = self.GetPosition()

        # Electron and positron direction vectors
        d1Dir = pos1 - vertex_position
        d2Dir = pos2 - vertex_position

        # Normalizing the direction vectors
        d1Dir /= np.linalg.norm(d1Dir)
        d2Dir /= np.linalg.norm(d2Dir)

        # Construct initial photon direction from theta and phi
        init_dir = initial_vector_function(theta, phi)

        if ref_direction is None:
            ref_direction = "RelativeX" # default to RelativeX if not specified

        # Project reference axis into photon frame
        refDir = ref_map[ref_direction]
        refDir_lab = refDir - np.dot(refDir, init_dir) * init_dir # Remove componenet along init_dir
        refDir_lab /= np.linalg.norm(refDir_lab)  # Normalize

        # Create second reference vector orthogonal to both
        ref2Dir = np.cross(init_dir, refDir_lab)
        ref2Dir /= np.linalg.norm(ref2Dir)

        # Compute azimuthal angles of each track in rotated frame
        phi1 = np.atan2(np.dot(ref2Dir, d1Dir), np.dot(refDir_lab, d1Dir)) + 2*np.pi
        phi2 = np.atan2(np.dot(ref2Dir, d2Dir), np.dot(refDir_lab, d2Dir)) + 2*np.pi

        phi = (phi1 + phi2) / 2
        if abs(phi1 - phi2) > np.pi:
            phi += np.pi
        
        # Wrap to [-pi, pi]
        return np.atan2(np.sin(phi), np.cos(phi))
    
    def ComputeIncomingGammaDirection(self):
        '''
        Reconstruct the photon direction using hit energy as well as track vectors (as done in MEGAlib reconstruction, 
        see the file: megalib/global/src/MPairEvent.cxx... this gets called later in MRERawEvent.cxx under the Validate() object)
        
        Note: when applied this is computing the direction for a singular event, not the overall reconstructed direction. 
                This should be averaged for comparison to MC truth.
        ------------------------------------------------------------------------------
        Parameters:
            None (uses the track directions and energies assigned to the vertex object)
        
        Returns:
            gamma_dir: unit vector of the reconstructed incoming gamma-ray direction
        '''

        if self.electron_dir is None or self.positron_dir is None:
            raise ValueError("Track directions not set.")

        if self.electron_energy is None or self.positron_energy is None:
            raise ValueError("Track energies not set.")

        d1 = self.electron_dir / np.linalg.norm(self.electron_dir)
        d2 = self.positron_dir / np.linalg.norm(self.positron_dir)

        E1 = self.electron_energy
        E2 = self.positron_energy

        gamma_dir = -(E1 * d1 + E2 * d2) / (E1 + E2) # energy-weighted reconstruction 
        gamma_dir /= np.linalg.norm(gamma_dir)

        self.gamma_dir = gamma_dir

        return gamma_dir

'''
(b) STANDALONE FUNCTIONS
'''
def initial_vector_function(theta, phi):
    '''
    Computes the vector for the initial direction of the incoming gamma-ray using theta and phi (this is MC info)
    ---------------------------------------------------------------
    Parameters:
        theta (float): Off-axis angle theta in degrees
        phi (float): Input phi angle in degrees

    Returns:
        np.ndarray: init_dir unit direction vector
    '''

    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    init_dir = np.array([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad)
    ])

    init_dir /= np.linalg.norm(init_dir)

    return init_dir

def project_to_layer(init_pos, init_dir, z_target):
    '''
    Projects a virtual point along a direction vector to a plane at z = z_target
    --------------------------------------------------------------
    Parameters:
        init_pos (np.ndarray): Initial position [x, y, z]
        init_dir (np.ndarray): Unit direction vector [dx, dy, dz]
        z_target (float): z-position of the detector layer

    Returns:
        np.ndarray: Projected point on the z = z_target plane
    '''

    t = (z_target - init_pos[2]) / init_dir[2] 
    projected_point = init_pos + t * init_dir
    return projected_point

def distance_to_virtual(position, virtual_point):
    '''
    Computes distance from a hit to a virtual projected point
    --------------------------------------------------------------
    Parameters:
        position (np.ndarray): Position of the hit [x, y, z]
        virtual_point (np.ndarray): Virtual projection point [x, y, z]

    Returns:
        float: Distance in cm from the hit to the virtual point
    '''

    distance = np.linalg.norm(position - virtual_point)
    return distance

def select_two_closest_hits(hits, projected_point):
    '''
    Finds the two hits closest to the projected virtual point within a given distance cutoff (currently set as 5cm)
    --------------------------------------------------------------
    Parameters:
        hits (list): List of RESE hit objects
        projected_point (np.ndarray): Virtual point from vertex projection

    Returns:
        tuple: (hit1, hit2, filtered_hits) where hit1 and hit2 are the closest hits (or None), and filtered_hits 
                is a list of both (if found)
    '''

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
    '''
    Plots histogram of distances between each hit and the virtual point
    --------------------------------------------------------------
    Parameters:
        hits (list of arrays): List of hit position arrays [x, y, z]
        projected_point (np.ndarray): Virtual projection point [x, y, z]
        bins (int): Number of bins in histogram
    
    Returns:
        None (displays a histogram plot)
    '''

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
    '''
    Identifies all hits in the detector layer directly below the vertex.
    --------------------------------------------------------------
    Parameters:
        event_id (int): Event ID to filter hits by
        vertex_z (float): z-position of the vertex
        HTXPosition, HTYPosition, HTZPosition (list): HT position data (in same units as vertex)
        layer_thickness (float): Vertical spacing between layers [cm]

    Returns:
        list: List of 3D hit position arrays [x, y, z] in the layer below
    '''

    # Range of z values to look in (1 layer)
    target_z = vertex_z - layer_thickness
    tolerance = 1e-6 # cm (only to handle floating point issues)

    hits_in_layer = []

    xs = HTXPosition.get(event_id, [])
    ys = HTYPosition.get(event_id, [])
    zs = HTZPosition.get(event_id, [])

    for x, y, z in zip(xs, ys, zs):
        if abs(z - target_z) < tolerance:
            hits_in_layer.append(np.array([x, y, z]))

    return hits_in_layer

'''
(c) VERTEXFINDER CLASS
'''
class VertexFinder:
    def __init__(self, Geometry, SearchRange = 30, NumberOfLayers = 2): # Let the default NumberOfLayers be 2

        self.Geometry = Geometry
        self.SearchRange = SearchRange
        self.NumberOfLayers = NumberOfLayers
        self.DetectorList = [M.MDStrip2D] # AstroPix detectors only
        self.two_hit_after_dead_material_counter = 0
        self.two_hit_in_two_layers_after_dead_material_counter = 0

    def IsInTracker(self, RESE):
        '''
        Determine if a RESE is part of the tracker
        ---------------------------------------------------------------
        Parameters:
            RESE (MRESE): The RESE to check

        Returns:
            bool: True if RESE is in the tracker, False otherwise
        '''

        if RESE.GetType() == M.MRESE.c_Track:
            return True

        VS = RESE.GetVolumeSequence()
        Detector = VS.GetDetector()

        if Detector.__class__ in self.DetectorList: 
            return True

        return False
    
    def best_hit_pairing(self, A1, A2, B1, B2):
        '''
        Given two hits in the layer below the vertex (B1, B2) and two hits in the layer above (A1, A2), determine the best pairing
        ----------------------------------------------------------------
        Parameters:
            A1, A2 (np.ndarray): 3D positions of the two hits in the layer above the vertex
            B1, B2 (np.ndarray): 3D positions of the two hits in the layer below the vertex
        
        Returns:
            tuple: ((A1, B1), (A2, B2)) or ((A1, B2), (A2, B1)) depending on which pairing has the smaller total distance
        '''
        d11 = np.linalg.norm(A1-B1)
        d12 = np.linalg.norm(A1-B2)
        d21 = np.linalg.norm(A2-B1)
        d22 = np.linalg.norm(A2-B2)

        pairing1 = d11+d22
        pairing2 = d12+d21

        if pairing1 <= pairing2:
            return (A1, B1),(A2, B2)
        else:
            return (A1, B2),(A2, B1)

    # There is a lot of background that goes into the following function - see the code documentation for a description
    def calculating_vertex_position(self, p1, v1, p2, v2):
        '''
        Given two tracks defined by points p1 and p2 and direction vectors v1 and v2, calculate the point of closest 
        approach between the two tracks
        ----------------------------------------------------------------
        Parameters:
            p1, p2 (np.ndarray): Points on each track (x, y, z)
            v1, v2 (np.ndarray): Direction vectors of each track
        
        Returns:
            np.ndarray: 3D point representing the reconstructed vertex position (midpoint of closest approach)
        '''

        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)

        w0 = p1 - p2
        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        d = np.dot(v1, w0) 
        e = np.dot(v2, w0)

        denom = a*c - b*b
        if abs(denom) < 1e-6:
            return None  # if very close to zero they are parallel

        t = (b*e - c*d)/denom
        s = (a*e - b*d)/denom

        pca1 = p1 + t*v1
        pca2 = p2 + s*v2

        return 0.5 * (pca1 + pca2)

    def FindVertices(self, RE, theta, phi):
        # Note that the basic pair event reconstruction follows the logic implemented in the Revan code. Additional logic has been added.
        '''
        Main method to find vertex candidates in an event
        ---------------------------------------------------------------
        Parameters:
            RE (MRESEvent): The reconstructed event object
            theta (float): Incoming photon polar angle (deg)
            phi (float): Incoming photon azimuthal angle (deg)

        Returns:
            list: List of Vertex objects found in the event
        '''

        Vertices = []

        # Filter RESEs to those in tracker with energy > 0
        RESEs = [
            RE.GetRESEAt(i) for i in range(RE.GetNRESEs())
            if self.IsInTracker(RE.GetRESEAt(i)) and RE.GetRESEAt(i).GetEnergy() > 0
        ]

        # Sorting by depth in the tracker: shallowest -> deepest (larger z-position value is shallower)
        RESEs.sort(key=lambda rese: rese.GetPosition().Z(), reverse=True)

        vertex_created_for_event = False # determine whether or not a vertex was assigned to an event

        for candidate in RESEs:
            # Determine if the hit is the only one in that layer
            OnlyHitInLayer = True
            for rese in RESEs:
                if rese == candidate:
                    continue
                if self.Geometry.AreInSameLayer(candidate, rese):
                    OnlyHitInLayer = False
                    break
            if not OnlyHitInLayer:
                continue 

            NBelow = [0] * self.SearchRange
            NAbove = [0] * self.SearchRange

            for rese in RESEs:
                if rese == candidate:
                    continue
                Distance = self.Geometry.GetLayerDistance(candidate, rese)
                if 0 < Distance < self.SearchRange:
                    NAbove[Distance] += 1
                elif Distance < 0 and abs(Distance) < self.SearchRange:
                    NBelow[abs(Distance)] += 1
            
            # Looking for the vertex below ("inverted V")
            if NAbove[1] != 0:
                continue

            StartIndex = 0
            StopIndex = 0
            LayersWithAtLeastTwoHitsBetweenStartAndStop = 0

            for Distance in range(1, self.SearchRange-1):
                if NBelow[Distance] == 0 and NBelow[Distance+1] == 0:
                    break
                StopIndex = Distance

                if StartIndex == 0 and NBelow[Distance] > 1 and NBelow[Distance+1] > 1:
                    StartIndex = Distance
                    
                if StartIndex != 0 and NBelow[Distance] >= 2:
                    LayersWithAtLeastTwoHitsBetweenStartAndStop += 1

            for Distance in range(StopIndex, 2, -1):
                if NBelow[Distance-1] >= 2 and NBelow[Distance-2] >= 2:
                    break
                StopIndex = Distance

            interlayerdistance = 1.5 #cm (change depending on detector geometry, this is for AMEGO-X)

            if LayersWithAtLeastTwoHitsBetweenStartAndStop < self.NumberOfLayers:
                continue

            # Search following N layers for first layer with 2+ hits
            selected_layer_hits = None
            selected_distance = None

            SearchLayers = 5 # CHANGE THIS FOR DESIRED NUMBER OF LAYERS BELOW CANDIDATE BEFORE 2+ HITS REQUIREMENT IS ENFORCED (new logic)

            for Distance in range(1, SearchLayers+1):
                layer_hits = [rese for rese in RESEs if self.Geometry.GetLayerDistance(candidate, rese) == -Distance]

                if len(layer_hits) >= 2:
                    selected_layer_hits = layer_hits
                    selected_distance = Distance
                    break

            if selected_layer_hits is None:
                continue

            # Project along MC direction to selected layer
            init_pos = np.array([candidate.GetPosition().X(),
                candidate.GetPosition().Y(),
                candidate.GetPosition().Z()])

            init_dir = initial_vector_function(theta, phi)

            # Move to the identified layer below candidate vertex and project virtual point on that layer
            z_target = candidate.GetPosition().Z() - selected_distance * interlayerdistance
            projected_point = project_to_layer(init_pos, init_dir, z_target)

            # Choose best two hits in that layer
            hit1, hit2, filtered_hits = select_two_closest_hits(selected_layer_hits, projected_point)

            if hit1 is None or hit2 is None:
                continue

            # Declare vertex
            vtx = Vertex(candidate, self.Geometry, [hit1, hit2])
            vtx.EventID = RE.GetEventID()
            Vertices.append(vtx)
            vertex_created_for_event = True # state that the event has been assigned a vertex

            '''
            FINALLY, compute the electron/positron track directions for gamma-ray reconstruction
            '''
            if vtx is not None:
                electron_track = np.array([hit1.GetPosition().X(),
                                        hit1.GetPosition().Y(),
                                        hit1.GetPosition().Z()]) \
                                - np.array([vtx.x, vtx.y, vtx.z])
                positron_track = np.array([hit2.GetPosition().X(),
                                        hit2.GetPosition().Y(),
                                        hit2.GetPosition().Z()]) \
                                - np.array([vtx.x, vtx.y, vtx.z])

                # Normalize directions
                vtx.electron_dir = electron_track / np.linalg.norm(electron_track)
                vtx.positron_dir = positron_track / np.linalg.norm(positron_track)

                vtx.electron_energy = hit1.GetEnergy()
                vtx.positron_energy = hit2.GetEnergy()

                vtx.gamma_dir = vtx.ComputeIncomingGammaDirection()
                vtx.vertex_type = 'type_1ht2ht' # assigning a type to the event for histogramming purposes 
                true_dir = initial_vector_function(theta, phi)
            
        '''
        SOME NEW LOGIC STARTS HERE
        '''
        # New logic for events that did not have a vertex assigned
        if not vertex_created_for_event:
            
            # Identify the topmost layer in the event
            shallowest_z = max(rese.GetPosition().Z() for rese in RESEs)

            hits_in_first_layer = [rese for rese in RESEs if abs(rese.GetPosition().Z()-shallowest_z) < 0.01] # in same layer as the shallowest hit

            if len(hits_in_first_layer) == 2:

                self.two_hit_after_dead_material_counter += 1

                hit1, hit2 = hits_in_first_layer

                if not self.Geometry.AreInSameLayer(hit1, hit2):
                    print("Z-close but different layer:", RE.GetEventID())

                # Now look for hits in the layer immediately below
                hits_in_layer_below = [rese for rese in RESEs
                    if self.Geometry.GetLayerDistance(hit1, rese) == -1]

                # For now also require exactly two hits in the layer below
                if len(hits_in_layer_below) == 2:

                    self.two_hit_in_two_layers_after_dead_material_counter += 1

                    hit1lay1, hit2lay1 = hits_in_first_layer
                    hit1lay2, hit2lay2 = hits_in_layer_below

                    A1 = np.array([hit1lay1.GetPosition().X(),
                        hit1lay1.GetPosition().Y(),
                        hit1lay1.GetPosition().Z()])

                    A2 = np.array([hit2lay1.GetPosition().X(),
                        hit2lay1.GetPosition().Y(),
                        hit2lay1.GetPosition().Z()])

                    B1 = np.array([hit1lay2.GetPosition().X(),
                        hit1lay2.GetPosition().Y(),
                        hit1lay2.GetPosition().Z()])

                    B2 = np.array([hit2lay2.GetPosition().X(),
                        hit2lay2.GetPosition().Y(),
                        hit2lay2.GetPosition().Z()])

                    # Choose best hit pairing
                    (track1, track2) = self.best_hit_pairing(A1, A2, B1, B2)

                    (p1, q1), (p2, q2) = track1, track2 # p = point in layer 1, q = point in layer 2

                    v1 = q1-p1
                    v2 = q2-p2

                    # Assigning the vertex a 3D position
                    vertex_point = self.calculating_vertex_position(p1, v1, p2, v2)
                    
                    if vertex_point is None:
                        return Vertices # skip
                    
                    vtx = Vertex(rese=None, Geometry=self.Geometry, AllRESEs=[hit1lay1, hit2lay1, hit1lay2, hit2lay2], position=vertex_point)
                    
                    '''
                    ADDED DIRECTION AND ENERGIES
                    '''
                    vtx.electron_dir = v1
                    vtx.positron_dir = v2

                    # Normalize directions
                    vtx.electron_dir /= np.linalg.norm(vtx.electron_dir)
                    vtx.positron_dir /= np.linalg.norm(vtx.positron_dir)

                    vtx.electron_energy = hit1lay1.GetEnergy() + hit1lay2.GetEnergy()
                    vtx.positron_energy = hit2lay1.GetEnergy() + hit2lay2.GetEnergy()

                    gamma_dir = vtx.ComputeIncomingGammaDirection()
                    vtx.gamma_dir = gamma_dir

                    true_dir = initial_vector_function(theta, phi)

                    #print(f"Event {RE.GetEventID()}: Reconstructed gamma direction: {gamma_dir}, True gamma direction: {true_dir}, Angle between: {np.arccos(np.clip(np.dot(gamma_dir, true_dir), -1.0, 1.0)) * 180/np.pi:.2f} degrees")
                    Vertices.append(vtx)

                    vtx.vertex_type = 'type_2ht2ht' # assigning a type to the event for histogramming purposes

        '''
        ENDS HERE
        '''

        '''
        If using the print statements below, this script must be run on a SIM file containing information only about events
        that convert in dead material. Run DeadMaterialConversionEventSelection.py to get a SIM file containing this information.
        '''
        #print("Number of events with two hits after dead material conversion:", self.two_hit_after_dead_material_counter)
        #print("Number of events with two hits in two layers after dead material conversion:", self.two_hit_in_two_layers_after_dead_material_counter)
        
        return Vertices

    def TopVertex(self, vertex_list):
        '''
        Select the top-most vertex (minimum z value) from a list of vertex candidates
        ---------------------------------------------------------------
        Parameters:
            vertex_list (list): List of Vertex objects

        Returns:
            Vertex: The vertex with the highest Z position
        '''

        return min(vertex_list, key=lambda v: v.GetZPosition())        

'''
(d) EVENTPLOTTING CLASS
'''
class EventPlotting:
    '''
    Handles plotting and data extraction from simulated and reconstructed gamma-ray events.
    Used for comparing true (simulated) and reconstructed hits and vertices.
    '''

    def __init__(self, inputfile, Geometry):
        self.inputfile = inputfile
        self.Geometry = Geometry

    def GetSimHits(self):
        '''
        Extract simulated HT hits from a .sim.gz file
        --------------------------------------------------------------
        Parameters:
            None (uses self.inputfile for the path to the SIM file)

        Returns:
            tuple of dicts: (HTXPosition, HTYPosition, HTZPosition) per event ID
        '''

        # Initializing dictionaries to hold hit positions for each event ID
        HTXPosition = {}
        HTYPosition = {}
        HTZPosition = {}
        ElectronHTX = {}
        ElectronHTY = {}
        ElectronHTZ = {}
        PositronHTX = {}
        PositronHTY = {}
        PositronHTZ = {}

        CurrentEventID = None
        ExpectEventID = False

        # Assigning event ID keys to the dictionaries
        HTXPosition[CurrentEventID] = []
        HTYPosition[CurrentEventID] = []
        HTZPosition[CurrentEventID] = []

        ElectronHTX[CurrentEventID] = []
        ElectronHTY[CurrentEventID] = []
        ElectronHTZ[CurrentEventID] = []

        PositronHTX[CurrentEventID] = []
        PositronHTY[CurrentEventID] = []
        PositronHTZ[CurrentEventID] = []

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


                                ElectronHTX[CurrentEventID] = []
                                ElectronHTY[CurrentEventID] = []
                                ElectronHTZ[CurrentEventID] = []

                                PositronHTX[CurrentEventID] = []
                                PositronHTY[CurrentEventID] = []
                                PositronHTZ[CurrentEventID] = []
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
                    
                    interactionid = list(map(int, line.split(';')[6:]))
                    if 3 in interactionid:
                        ElectronHTX[CurrentEventID].append(XPositionColumn)
                        ElectronHTY[CurrentEventID].append(YPositionColumn)
                        ElectronHTZ[CurrentEventID].append(ZPositionColumn)
                    if 4 in interactionid:
                        PositronHTX[CurrentEventID].append(XPositionColumn)
                        PositronHTY[CurrentEventID].append(YPositionColumn)
                        PositronHTZ[CurrentEventID].append(ZPositionColumn)

        return (HTXPosition, HTYPosition, HTZPosition, ElectronHTX, ElectronHTY, ElectronHTZ, PositronHTX, PositronHTY, PositronHTZ) 
    
    def GetRESEEvents(self, Geometry, inputfile, MaxNumberOfEvents=None):
        '''
        Extract clustered and noised RESE hits from a SIM file using MEGAlib evta reader
        --------------------------------------------------------------
        Parameters:
            Geometry (MGeometry): Detector geometry object
            inputfile (str): Path to the SIM file
            MaxNumberOfEvents (int, optional): Stop after this many events

        Returns:
            dict: {EventID: [(X, Y, Z, marker, label)]}
        '''

        # Getting the noised and clustered events
        Reader = M.MFileEventsEvta(self.Geometry)
        Reader.Open(M.MString(self.inputfile))

        Clusterizer = M.MERHitClusterizer()
        Clusterizer.SetParameters(0,0,0,0,0,0,0,0,True) # Default clustering parameters are (1,-1)...CURRENT INPUT STOPS CLUSTERING

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

    def PlottingSimAndRESEs(self, EventID, EHTX, EHTY, EHTZ,  PHTX, PHTY, PHTZ, RESEHits, Vertices, MCPoints=None):
        '''
        3D plot of simulated (MC) hits (electron and positron), RESE hits, identified vertices, and MC interaction 
        point (if provided) for a given event ID
        --------------------------------------------------------------
        Parameters:
            EventID (int): ID of the event to plot
            EHTX, EHTY, EHTZ (dict): Simulated electron hit positions
            PHTX, PHTY, PHTZ (dict): Simulated positron hit positions
            RESEHits (dict): RESE hits from reconstruction (either clustered or not clustered)
            Vertices (list): Vertex objects for this event
            MCPoints (dict, optional): True MC interaction point per event
        '''

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Event ID {EventID} : MC Hits and RESEs")
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        ax.set_zlabel("Z [cm]")

        '''
        UNCOMMENT THIS SECTION TO SHOW MC HITS (without e- / e+ distinction)
        if EventID in HTX and HTX[EventID]:
            ax.scatter(HTX[EventID], HTY[EventID], HTZ[EventID], c='blue', marker='o', s=20, alpha=0.3, label='MC HT Hits')
        '''

        # Plot simulated tracker hits amd color based on electron and positron hits
        if EventID in EHTX and EHTX[EventID]:
            ax.scatter(EHTX[EventID], EHTY[EventID], EHTZ[EventID], c='purple', marker='*', s=40, alpha=0.4, label='Electron HT')

        if EventID in PHTX and PHTX[EventID]:
            ax.scatter(PHTX[EventID], PHTY[EventID], PHTZ[EventID], c='orange', marker='^', s=40, alpha=0.6, label='Positron HT')

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
        #RedClusterLegend = Line2D([0], [0], marker='s', color='red', linestyle='None', markersize=8, label='Clustered RESEs') UNCOMMENT IF RUNNING CLUSTERING
        RedHTLegend = Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=8, label='Hit RESEs')
        ElectronHTLegend = Line2D([0], [0], marker='*', color='purple', linestyle='None', markersize=8, label='Electron HT')
        PositronHTLegend = Line2D([0], [0], marker='^', color='orange', linestyle='None', markersize=8, label='Positron HT')
        #BlueLegend = Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=8, label='MC Hits') UNCOMMENT TO SHOW MC HITS (without e- / e+ distinction)
        VertexLegend = Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Identified Vertex Position(s)')
        MCPointLegend = Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=10, label='MC Vertex')

        # Combine into final legend list depending on what's present
        legend_handles = []
        if EventID in RESEHits and RESEHits[EventID]:
            legend_handles += [RedHTLegend]
        if Vertices:
            legend_handles += [VertexLegend]
        if MCPoints is not None and EventID in MCPoints:
            legend_handles += [MCPointLegend]
            legend_handles += [ElectronHTLegend, PositronHTLegend]
        
        ax.legend(handles=legend_handles, loc='upper right')
    
        plt.tight_layout()
        plt.show()
    
    def PlotGammaRayReconstructionHistogram(self, all_angle_differences, oneht_twoht_angles=None, twoht_twoht_angles=None, revan_angles=None): # , revan_angle_differencess
        '''
        Plot histogram of angle differences between reconstructed and true gamma-ray directions
        --------------------------------------------------------------
        Parameters:
            angle_differences (list): List of angle differences (between reconstructed and actual) in degrees
            inputfile (str): filename
        
        Returns:
            None (displays a histogram plot)
        '''

        all_values = all_angle_differences.copy()
        if oneht_twoht_angles:
            all_values += oneht_twoht_angles
        if twoht_twoht_angles:
            all_values += twoht_twoht_angles
        if revan_angles:
            all_values += revan_angles

        binwidth = 1
        bins = np.arange(min(all_values), max(all_values) + binwidth, binwidth)

        # Plot each histogram
        # plt.hist(all_angle_differences, bins=bins, alpha=1, linewidth=2, edgecolor='black', cumulative=True, density=True, histtype='step', label='All reconstructed events (current logic)')

        if oneht_twoht_angles:
            plt.hist(oneht_twoht_angles, bins=bins, alpha=1, linewidth=2, edgecolor='blue', cumulative=True, density=True, histtype='step', label='1 hit 2 hit')

        if twoht_twoht_angles:
            plt.hist(twoht_twoht_angles, bins=bins, alpha=1, linewidth=2, edgecolor='red', cumulative=True, density=True, histtype='step', label='2 hit 2 hit')
            
        if revan_angles:
            plt.hist(revan_angle_differences, bins=bins, alpha=1, linewidth=2, edgecolor='green', cumulative=True, density=True, histtype='step', label='All Revan reconstructed events')

        plt.xlabel("Angle difference between reconstructed and true gamma-ray direction [degrees]")
        plt.ylabel("Normalized # events")
        plt.title("Gamma-Ray Reconstruction Accuracy")
        plt.legend()
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.show()
    
    def PlotVertexHistogram(self, NumberOfVerticesPerEvent, inputfile):
        '''
        Plot histogram of the number of vertices per event
        --------------------------------------------------------------
        Parameters:
            NumberOfVerticesPerEvent (list): List of vertex counts per event
            inputfile (str): Filename prefix for saving the plot
        '''

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

'''
(e) MCINTERACTION CLASS
'''
class MCInteraction: # does not NEED to be a class -> this is mostly just for organizational purposes
    def GetMCInteractionPoints(inputfile):
        '''
        Extracts the true interaction points from a .sim.gz file
        --------------------------------------------------------------
        Parameters:
            inputfile (str): Path to the input .sim.gz file

        Returns:
            dict: Dictionary mapping EventID to its true MC interaction point as a NumPy array [x, y, z]
        '''

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
        '''
        Plots residuals between identified vertex positions and true MC interaction points
        --------------------------------------------------------------
        Parameters:
            MCPoints (dict): Dictionary of true MC points from `get_mc_interaction_points`
            VertexDict (dict): Dictionary mapping EventID to list of identified Vertex objects
            EventNumberToEventID (dict, optional): Mapping the event number to the event ID
        
        Returns:
            None (displays histogram and scatter plots of residuals)
        '''

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
        plt.hist(Residuals, bins=np.arange(0, 100, 1.5), edgecolor='black')
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
        plt.hist(dz_list, bins=np.arange(-50, 50, 1.5), edgecolor='black')
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
MAIN FUNCTION FOR EXECUTION IN TERMINAL
'''
if __name__ == "__main__":

    # Load in and initialize MEGAlib 
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    G = M.MGlobal()
    G.Initialize()

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Process a .sim.gz file from MEGAlib Cosima output.")
    parser.add_argument("inputfile", help="Enter path to desired input file.")
    parser.add_argument("--plot-event-number", type=int, default=None, help="Enter the desired event number (not event ID) to be plotted.")
    parser.add_argument("--plot-eventID", type=int, default = None, help="Enter the desired event ID (not event number) to be plotted.  ")
    parser.add_argument('--layers', type=int, default=2, help='Enter desired number of layers with 2+ hits to classify vertex (for one hit, two hit morphology). Default is 2.')
    parser.add_argument('--plot-histogram', action='store_true', help='Plot number of vertices per event histogram.')
    parser.add_argument('--plot-residuals', action='store_true', help='Plot residuals between MC and reconstructed vertex.')
    parser.add_argument('--plot-events', action='store_true', help='Plot individual event visualizations.')
    parser.add_argument('--ref-dir', type=str, default='RelativeX', choices=['RelativeX', 'RelativeY', 'RelativeZ'], help="Reference direction used for azimuthal angle calculation (default: RelativeX)")
    parser.add_argument('--theta', type=float, default=0.0, help='Polar angle theta (in degrees) of the incoming gamma-ray (default: 0 degrees)')
    parser.add_argument('--phi', type=float, default=0.0, help='Azimuthal angle phi (in degrees) of the incoming gamma-ray (default: 0 degrees)')
    parser.add_argument('--plot-distance-dist', action='store_true', help='Plot distance distribution of hits relative to virtual hit in the layer below vertex')
    parser.add_argument('--gr-hist', action='store_true', help='Plot histogram of angle differences between reconstructed and true gamma-ray directions')

    # Loading in the required geometry (AMEGO-X for this analysis, can change to desired geometry)
    GeometryName = "../Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
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
    Clusterizer.SetParameters(0,0,0,0,0,0,0,0,True) # can change back to default if needed
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

    EP = EventPlotting(inputfile, Geometry)
    HTX, HTY, HTZ, ElectronHTX, ElectronHTY, ElectronHTZ, PositronHTX, PositronHTY, PositronHTZ = EP.GetSimHits()

    all_vertices = []
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
        all_vertices.extend(Vertices)

        if Vertices:
            top_vertex = VF.TopVertex(Vertices)
            vertex_z = top_vertex.GetPosition()[2]

            hits_below = get_hits_in_layer_below(RE.GetEventID(), vertex_z, HTX, HTY, HTZ)

            # Projected point -> make function
            init_dir = initial_vector_function(args.theta, args.phi)
            
            init_pos = np.array([
                top_vertex.GetPosition()[0],
                top_vertex.GetPosition()[1],
                vertex_z
            ])

            z_target = vertex_z - 1.5 # cm (interlayer distance)
            projected_point = project_to_layer(init_pos, init_dir, z_target)

            '''
            if args.plot_distance_dist:
                # Plot distribution 
                for hit in hits_below:
                    distance = distance_to_virtual(hit, projected_point)
                    all_distances_to_virtual.append(distance)

                    if distance < 5.0:
                        passing_distances.append(distance)
            else: 
                pass
            '''
            
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

    # Compute true direction
    true_dir = initial_vector_function(args.theta, args.phi)

    all_angle_differences = []
    oneht_twoht_angles = []
    twoht_twoht_angles = []

    for vtx in all_vertices:
        cos_angle = np.clip(np.dot(vtx.gamma_dir, true_dir), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        all_angle_differences.append(angle_deg)

        if hasattr(vtx, "vertex_type"):
            if vtx.vertex_type == "type_1ht2ht":
                oneht_twoht_angles.append(angle_deg)
            elif vtx.vertex_type == "type_2ht2ht":
                twoht_twoht_angles.append(angle_deg)

    # Compute mean reconstructed direction
    if all_vertices:
        mean_dir = np.mean([vtx.gamma_dir for vtx in all_vertices], axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
    else:
        mean_dir = None

    # plotting the histogram for reconstructed gamma rays  
    if args.gr_hist:
        EP.PlotGammaRayReconstructionHistogram(all_angle_differences, oneht_twoht_angles, twoht_twoht_angles)
        print("\n--------- OVERALL RECONSTRUCTION PERFORMANCE ---------")
        print("Total # of reconstructed events:", len(all_vertices))
        print("Average reconstructed direction:", mean_dir)
        print("True direction (from MC input):", true_dir)
        print("Median angle between reconstructed and MC truth:", f"{np.median(all_angle_differences):.3f} degrees")

        # PERCENTILE CALCULATIONS
        allpercentile68 = np.percentile(all_angle_differences, 68) 
        print(f"68th percentile angle difference for all reconstructed events: {allpercentile68:.3f} degrees")
        # -----------------------------
        if len(oneht_twoht_angles) > 0:
            oneht2htpercentile68 = np.percentile(oneht_twoht_angles, 68) 
            print(f"68th percentile angle difference for 1 hit 2 hit events: {oneht2htpercentile68:.3f} degrees")
        else:
            oneht2htpercentile68 = None
            print("No 1 hit 2 hit events found; cannot compute 68th percentile.")
        # -----------------------------
        if len(twoht_twoht_angles) > 0:
            twoht2htpercentile68 = np.percentile(twoht_twoht_angles, 68)
            print(f"68th percentile angle difference for 2 hit 2 hit events: {twoht2htpercentile68:.3f} degrees")
        else:
            twoht2htpercentile68 = None
            print("No 2 hit 2 hit events found; cannot compute 68th percentile.")
        
        #print(f"68th percentile angle difference for 2 hit 2 hit events: {twoht2htpercentile68:.3f} degrees")

        #revanpercentile68 = np.percentile(revan_angle_differences, 68) 

    # Write out the leftover azimuthal angles into the output file
    if event_id_list:
        with open(output_phi_filename, 'a') as f:
            for eid, phi in zip(event_id_list, phi_values):
                f.write(f"{eid} {phi}\n")

    if args.plot_distance_dist and all_distances_to_virtual:
        # Plot distribution 
        for hit in hits_below:
            distance = distance_to_virtual(hit, projected_point)
            all_distances_to_virtual.append(distance)

            if distance < 5.0:
                passing_distances.append(distance)

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

    if args.plot_histogram:
        EP.PlotVertexHistogram(NumberOfVerticesPerEvent, inputfile)

    if args.plot_events:
        print("Accessing simulated and RESE hits...")
        '''
        NOW THE EVENT PLOTTING FOR CHOSEN EVENT
        '''
        EP = EventPlotting(inputfile, Geometry=Geometry)
        # Go through the sim hits
        (HTX, HTY, HTZ, ElectronHTX, ElectronHTY, ElectronHTZ, PositronHTX, PositronHTY, PositronHTZ) = EP.GetSimHits()
        print(f"Parsed sim hits for {len(HTX)} events")

        # Go through RESEs
        RESEHits = EP.GetRESEEvents(Geometry, inputfile, MaxNumberOfEvents=None) # set some max number of events if desired
        print(f"Parsed RESEs for {len(RESEHits)} events")
        
        # Find shared event IDs once
        SharedEventID = sorted(set(HTX.keys()).intersection(RESEHits.keys()))
        print(f"Found {len(SharedEventID)} events with both simulated and clustered data")
        
        # Get MC points once
        MCPoints = MCInteraction.GetMCInteractionPoints(inputfile)
        print(f"Parsed MC interaction points for {len(MCPoints)} events")
        
        # Plot based on args
        if args.plot_eventID is not None:
            event_id_to_plot = args.plot_eventID
            if event_id_to_plot in SharedEventID:
                print(f"Plotting event with Event ID {event_id_to_plot}")
                EP.PlottingSimAndRESEs(event_id_to_plot, ElectronHTX, ElectronHTY, ElectronHTZ, PositronHTX, PositronHTY, PositronHTZ, RESEHits, VertexDict.get(event_id_to_plot, []), MCPoints=MCPoints)
            else:
                print(f"Event ID {event_id_to_plot} not found in data.")

        elif plot_event_number is not None:
            if plot_event_number in EventNumberToEventID:
                event_id_to_plot = EventNumberToEventID[plot_event_number]
                if event_id_to_plot in SharedEventID:
                    print(f"Plotting event number {plot_event_number} (Event ID {event_id_to_plot})")
                    EP.PlottingSimAndRESEs(event_id_to_plot, HTX, HTY, HTZ, RESEHits, VertexDict.get(event_id_to_plot, []), MCPoints=MCPoints)
                else:
                    print(f"Event ID {event_id_to_plot} has no RESE data.")
            else:
                print(f"Requested event number {plot_event_number} is out of bounds.")
        else:
            print("No specific event specified. Plotting all events.")
            for Event_ID in SharedEventID:
                EP.PlottingSimAndRESEs(Event_ID, ElectronHTX, ElectronHTY, ElectronHTZ, PositronHTX, PositronHTY, PositronHTZ, RESEHits, VertexDict.get(Event_ID, []), MCPoints=MCPoints)

    if args.plot_residuals:
        # Only parse MC points if not already done above
        if not args.plot_events:
            MCPoints = MCInteraction.GetMCInteractionPoints(inputfile)
        MCInteraction.PlotVertexResiduals(MCPoints, VertexDict, EventNumberToEventID)

'''
End of file
'''