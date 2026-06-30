'''
This file is separated into standalone functions and classes:
    (a) Vertex class: defines properties of the vertex (position, ID, adjacent hits in the layer below, associated
                                                        gamma ray direction, and the azimuthal angle)
    (b) GroupedHit class:

    (c) Functions:
        i. Uses input theta and phi (MC info) to compute the initial gamma ray direction vector
        ii. Projects the initial gamma ray direction vector onto the next layer
        iii. Calculates and returns the distance from a hit to the projected/virtual point
        iv. Finds the two hits closest to the identified virtual hit
        v. Plots the distribution of the distances from the hit to the virtual point
        vi. Gets the hits in the layer below the identified vertex
        vii.
        viii.
        ix.
        x.
    (d) VertexFinder class: locates the vertex - refer to the MERTrack.cxx file for the process of vertex identification. Also
                            reconstructs the gamma-ray direction using the identified vertex and hits in the layer below. 
    (e) EventPlotting class: contains structure to plot the MC hits, RESE hits, MC vertex/interaction point, and identified vertex
                            for a specified event
    (f) MCInteraction class: structure to extract the MC interaction points for each event
'''

# ------------------------------
# IMPORTING NECESSARY LIBRARIES AND MODULES

import ROOT as M
import gzip
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import numpy as np
import os
import gc
# ------------------------------

# Setting graphing font to Times New Roman (not important, personal choice)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]


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
            ref_direction = "RelativeX" # default to RelativeX if not specified -> change?

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
    
    def ComputePhi_RelativeX(vtx, photon_dir, hit1, hit2):
        k = np.asarray(photon_dir)
        if np.linalg.norm(k) == 0:
            return None
        k /= np.linalg.norm(k)

        x_hat = np.array([1.0, 0.0, 0.0]) # in detector frame
        e_pol = np.cross(k, x_hat) # this gives the RelativeX direction defined in MEGAlib
        e_pol /= np.linalg.norm(e_pol)

        e2 = np.cross(k, e_pol)  # creates an orthonormal basis for a plane perpendicular to the photon direction
        e2 /= np.linalg.norm(e2)

        # hit positions
        pos1 = np.array([hit1.GetPosition().X(),hit1.GetPosition().Y(),hit1.GetPosition().Z()])
        pos2 = np.array([hit2.GetPosition().X(),hit2.GetPosition().Y(),hit2.GetPosition().Z()])
        vpos = np.array([vtx.GetXPosition(),vtx.GetYPosition(),vtx.GetZPosition()])

        # direction vectors for the electron/positron hits relative to the identified vertex location
        d1 = pos1 - vpos
        d2 = pos2 - vpos
        if np.linalg.norm(d1) == 0 or np.linalg.norm(d2) == 0:
            return None
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        # project d1 and d2
        d1_perp = d1 - np.dot(d1, k) * k
        d2_perp = d2 - np.dot(d2, k) * k
        if np.linalg.norm(d1_perp) == 0 or np.linalg.norm(d2_perp) == 0:
            return None
        d1_perp /= np.linalg.norm(d1_perp)
        d2_perp /= np.linalg.norm(d2_perp)

        # azimuthal angle of each track 
        phi1 = np.arctan2(np.dot(d1_perp, e2), np.dot(d1_perp, e_pol))
        phi2 = np.arctan2(np.dot(d2_perp, e2), np.dot(d2_perp, e_pol))

        # bisector
        phi = (phi1 + phi2) / 2
        if abs(phi1 - phi2) > np.pi: # redundant
            phi += np.pi

        return np.arctan2(np.sin(phi), np.cos(phi))
    
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

        gamma_dir = -(E1* d1 + E2* d2)/(E1 + E2) # energy-weighted reconstruction
        gamma_dir /= np.linalg.norm(gamma_dir)

        self.gamma_dir = gamma_dir

        return gamma_dir


'''
(b) GroupedHit CLASS
'''
# making this a class for now since it is technically a "fake" hit with its own properties
class GroupedHit:
    '''
    New hit created by clustering other hits
    Contains the same properties of an RESE
    '''
    class CVec: # "clustered vector"
        '''
        was struggling with how to create access to properties like an RESE. this was a solution I found (nesting a helper class) is there a better way?
        '''
        def __init__(self, pos_array):
            self.posarr = pos_array
        def X(self): return float(self.posarr[0])
        def Y(self): return float(self.posarr[1])
        def Z(self): return float(self.posarr[2])

    def __init__(self, position, energy):
        '''
        Parameters:
            position (array): [x, y, z] midpoint of the merged hits
            energy (float): summed energy of the merged hits
        '''
        self._position = np.asarray(position, dtype=float)
        self._energy   = float(energy)

    def GetPosition(self):
        return GroupedHit.CVec(self._position)

    def GetEnergy(self):
        return self._energy

    def GetID(self):
        return -1  # meaningless id for the fake hit since it doesn't have one -> again, can make more robust


'''
(c) STANDALONE FUNCTIONS
'''
# (i)
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

# (ii)
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

# (iii)
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

# (iv)
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
        #print('rejected event due to distance cutoff')
        return None, None, []

    # Sort by distance and select the two closest
    sorted_hits = sorted(hits_with_dist, key=lambda x: x[1])
    hit1, hit2 = sorted_hits[0][0], sorted_hits[1][0]
    filtered_hits = [hit1, hit2]

    return hit1, hit2, filtered_hits

# (v)
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

# (vi)
def get_hits_in_layer_below(event_id, vertex_z, HTXPosition, HTYPosition, HTZPosition, interlayerdistance):
    '''
    Identifies all hits in the detector layer directly below the vertex.
    --------------------------------------------------------------
    Parameters:
        event_id (int): Event ID to filter hits by
        vertex_z (float): z-position of the vertex
        HTXPosition, HTYPosition, HTZPosition (list): HT position data (in same units as vertex)
        interlayerdistance (float): spacing between layers of the tracker 
    Returns:
        list: List of 3D hit position arrays [x, y, z] in the layer below
    '''

    # Range of z values to look in (1 layer)

    target_z = vertex_z - interlayerdistance
    tolerance = 1e-6 # cm (only to handle floating point issues)

    hits_in_layer = []

    xs = HTXPosition.get(event_id, [])
    ys = HTYPosition.get(event_id, [])
    zs = HTZPosition.get(event_id, [])

    for x, y, z in zip(xs, ys, zs):
        if abs(z - target_z) < tolerance:
            hits_in_layer.append(np.array([x, y, z]))

    return hits_in_layer

# (vii)
def clustering_hits_bydist(hits): # USING STRAIGHT UP DISTANCE COMPUTATION
    '''
    Take the two closest hits and merge them into one. If more than two hits remain after this,
    merge the closest two. Continue until exactly two hits remain and use those for reconstruction.
    The locations of the "fake" hits are determined by the midpoint of the clustered hits.
    --------------------------------------------------------------
    Parameters:
        hits (list): List of hit objects (RESEs or clustered)

    Returns:
        list: Exactly two hits, or an empty list if the
              input contains fewer than two hits.
    '''
    
    if len(hits) < 2:
        return [] # if there are less than two hits, return an empty list

    current = list(hits) # copy the input list to make modifications without changing original data

    while len(current) > 2: # perform whenever there are more than two hits found -- IF EXACTLY 2 LEAVE TO RECONSTRUCTION   
        min_dist = float('inf') # keep track of smallest distance between any two hits, start at infinity so first cluster is always saved
        index_i, index_j = -1, -1 # initialize indicies of closest pair found in this iteration

        # Find the closest pair
        for i in range(0, len(current)):
            point_i = np.array([current[i].GetPosition().X(),
                           current[i].GetPosition().Y(),
                           current[i].GetPosition().Z()])
            for j in range(i + 1, len(current)):
                point_j = np.array([current[j].GetPosition().X(),
                               current[j].GetPosition().Y(),
                               current[j].GetPosition().Z()])
                distance = np.linalg.norm(point_i - point_j)
                if distance < min_dist:
                    min_dist = distance
                    index_i, index_j = i, j

        hit_i = current[index_i]
        hit_j = current[index_j]

        point_i = np.array([hit_i.GetPosition().X(), hit_i.GetPosition().Y(), hit_i.GetPosition().Z()])
        point_j = np.array([hit_j.GetPosition().X(), hit_j.GetPosition().Y(), hit_j.GetPosition().Z()])

        combined = GroupedHit(
            position=(point_i + point_j) / 2.0, 
            energy=hit_i.GetEnergy() + hit_j.GetEnergy() # can i just add the energy?
        )

        # assign a new hit to the cluster
        current = [h for k, h in enumerate(current) if k not in (index_i, index_j)]
        current.append(combined)

    return current # has length of 2

# (viii)
def clustering_hits_by_angle(hits, hit_in_previous_layer):
    if len(hits) < 2:
        #print("less than 2 hits found--CHECK")  
        return []

    current = list(hits)

    max_angle = 30.0 # in degrees (best determined cutoff value)

    while len(current) > 2:
        min_angle = float('inf')
        index_i, index_j = -1, -1
        found_valid_pair = False

        for i in range(len(current)):
            point_i = np.array([
                current[i].GetPosition().X(),
                current[i].GetPosition().Y(),
                current[i].GetPosition().Z()
            ])
            dir_i = point_i - hit_in_previous_layer
            norm_i = np.linalg.norm(dir_i)
            if norm_i == 0:
                continue
            dir_i /= norm_i

            for j in range(i + 1, len(current)):
                point_j = np.array([
                    current[j].GetPosition().X(),
                    current[j].GetPosition().Y(),
                    current[j].GetPosition().Z()
                ])
                dir_j = point_j - hit_in_previous_layer
                norm_j = np.linalg.norm(dir_j)
                if norm_j == 0:
                    continue
                dir_j /= norm_j

                cosine = np.clip(np.dot(dir_i, dir_j), -1.0, 1.0) 
                angle = np.rad2deg(np.arccos(cosine))

                # only consider valid pairs
                if angle <= max_angle:
                    if angle < min_angle:
                        min_angle = angle
                        index_i, index_j = i, j
                        found_valid_pair = True

        # no valid pairs left under threshold so stop clustering
        if not found_valid_pair:
            return [] # skip the event if it doesn't meet requirements

        # merge best valid pair
        hit_i = current[index_i]
        hit_j = current[index_j]

        point_i = np.array([
            hit_i.GetPosition().X(),
            hit_i.GetPosition().Y(),
            hit_i.GetPosition().Z()
        ])
        point_j = np.array([
            hit_j.GetPosition().X(),
            hit_j.GetPosition().Y(),
            hit_j.GetPosition().Z()
        ])

        combined = GroupedHit(
            position=(point_i + point_j) / 2.0,
            energy=hit_i.GetEnergy() + hit_j.GetEnergy()
        )

        current = [h for k, h in enumerate(current) if k not in (index_i, index_j)]
        current.append(combined)

    return current

# (ix)
# CAN ENTIRELY REMOVE NOW
'''
def cluster_by_distance(hits, distance_threshold=0.05):  # tune threshold to be approximately the pixel size
    
    # Step 1: merge hits that are likely from the same particle
    # (adjacent pixel hits) based on spatial proximity
    
    current = list(hits)

    while len(current) > 2:
        min_dist = float('inf')
        index_i, index_j = -1, -1

        for i in range(0, len(current)):
            point_i = np.array([current[i].GetPosition().X(),
                                current[i].GetPosition().Y(),
                                current[i].GetPosition().Z()])
            for j in range(i + 1, len(current)):
                point_j = np.array([current[j].GetPosition().X(),
                                    current[j].GetPosition().Y(),
                                    current[j].GetPosition().Z()])
                dist = np.linalg.norm(point_i - point_j)
                if dist < min_dist:
                    min_dist = dist
                    index_i, index_j = i, j

        if min_dist > distance_threshold:
            break  # no more hits are close enough to be the same particle, stop clustering

        hit_i = current[index_i]
        hit_j = current[index_j]
        point_i = np.array([hit_i.GetPosition().X(), hit_i.GetPosition().Y(), hit_i.GetPosition().Z()])
        point_j = np.array([hit_j.GetPosition().X(), hit_j.GetPosition().Y(), hit_j.GetPosition().Z()])

        combined = GroupedHit(
            position=(point_i + point_j) / 2.0,
            energy=hit_i.GetEnergy() + hit_j.GetEnergy()
        )
        current = [h for k, h in enumerate(current) if k not in (index_i, index_j)]
        current.append(combined)

    return current
'''
# (x)
def select_best_pair(hits, vertex_pos):
    
    # Step 2: from the clustered hits, select the two with the
    # smallest opening angle as seen from the vertex
    
    if len(hits) < 2:
        return None, None
    if len(hits) == 2:
        return hits[0], hits[1]

    min_angle = float('inf')
    best_i, best_j = -1, -1

    for i in range(len(hits)):
        point_i = np.array([hits[i].GetPosition().X(),
                            hits[i].GetPosition().Y(),
                            hits[i].GetPosition().Z()])
        dir_i = point_i - vertex_pos
        norm_i = np.linalg.norm(dir_i)
        if norm_i == 0:
            continue
        dir_i /= norm_i

        for j in range(i + 1, len(hits)):
            point_j = np.array([hits[j].GetPosition().X(),
                                hits[j].GetPosition().Y(),
                                hits[j].GetPosition().Z()])
            dir_j = point_j - vertex_pos
            norm_j = np.linalg.norm(dir_j)
            if norm_j == 0:
                continue
            dir_j /= norm_j

            cosine = np.clip(np.dot(dir_i, dir_j), -1.0, 1.0)
            angle = np.rad2deg(np.arccos(cosine))

            if angle < min_angle:
                min_angle = angle
                best_i, best_j = i, j

    if best_i == -1:
        return None, None

    return hits[best_i], hits[best_j]


'''
(d) VERTEXFINDER CLASS
'''
class VertexFinder:
    def __init__(self, Geometry, SearchRange = 30, NumberOfLayers = 0): # Let the default NumberOfLayers be 0 (was originally 2)

        self.Geometry = Geometry
        self.SearchRange = SearchRange
        self.NumberOfLayers = NumberOfLayers
        self.DetectorList = [M.MDStrip2D] # AstroPix detectors only

        # Derive interlayer distance from the first detector matching DetectorList
        self.interlayer_distance = None
        for i in range(Geometry.GetNDetectors()):
            det = Geometry.GetDetectorAt(i)
            if det.__class__ in self.DetectorList:
                self.interlayer_distance = det.GetStructuralPitch().Z()
                #print(self.interlayer_distance)
                break

        if self.interlayer_distance is None:
            raise RuntimeError("No detector matching DetectorList found in geometry — cannot determine interlayer distance.")

        self.two_hit_after_dead_material_counter = 0 # this line and one below only apply if running on a filtered file containing only dead material conversion hits
        self.two_hit_in_two_layers_after_dead_material_counter = 0 

    def IsInTracker(self, RESE): # can look at comparing this to the megalib implementation to get closer to
        '''
        Determine if an RESE is part of the tracker
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
    
    def BestHitPairing(self, A1, A2, B1, B2):
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
    def CalculatingVertexPosition(self, p1, v1, p2, v2):
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
            #print("2ht 2ht tracks are parallel (or very close to)... cannot compute vertex position")
            return None  # if very close to zero they are parallel

        t = (b*e - c*d)/denom
        s = (a*e - b*d)/denom

        pca1 = p1 + t*v1
        pca2 = p2 + s*v2

        return 0.5 * (pca1 + pca2)

    def FindVertices(self, RE, theta, phi, polarization):
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

        # initialize counters to print debug statements
        nRejectedNotOnlyHit       = 0
        nRejectedHitAbove         = 0
        nRejectedLayerRequirement = 0
        nRejectedNo2HitLayer      = 0
        nRejectedClustering       = 0

        Vertices = []
        LayerRequirement = 4 # number of layers with 2+ hits required after the vertex (used for "event quality" selection), revan sets a minimum of 4
        
        # Filter RESEs to those in tracker with energy > 0
        RESEs = [RE.GetRESEAt(i) for i in range(RE.GetNRESEs())
            if self.IsInTracker(RE.GetRESEAt(i)) and RE.GetRESEAt(i).GetEnergy() > 0]

        # Sorting by depth in the tracker: shallowest -> deepest (larger z-position value is shallower)
        RESEs.sort(key=lambda rese: rese.GetPosition().Z(), reverse=True)

        vertex_created_for_event = False # flag to determine whether or not a vertex was assigned to an event

        # -------------------------------
        # FOR EVENTS WITH A 1HT 2HT MORPHOLOGY, apply the subsequent logic
        # -------------------------------
        for candidate in RESEs:
            OnlyHitInLayer = True
            for rese in RESEs:
                if rese == candidate: # skip if the hit is the candidate itself
                    continue
                if self.Geometry.AreInSameLayer(candidate, rese): # skip if there is another hit in the same layer as the candidate
                    OnlyHitInLayer = False # now this becomes false
                    break
            if not OnlyHitInLayer: # if there are multiple hits in the candidate layer, reject
                nRejectedNotOnlyHit += 1 # increase the counter for the event
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
                nRejectedHitAbove += 1
                continue
            
            StartIndex = 0
            StopIndex = 0
            LayersWithAtLeastTwoHitsBetweenStartAndStop = 0

            for Distance in range(1, self.SearchRange-1):
                if NBelow[Distance] == 0 and NBelow[Distance+1] == 0:
                    break
                StopIndex = Distance

                if StartIndex == 0 and NBelow[Distance] > 1:
                    StartIndex = Distance
                    
                if StartIndex != 0 and NBelow[Distance] >= 2:
                    LayersWithAtLeastTwoHitsBetweenStartAndStop += 1
            
            for Distance in range(StopIndex, 2, -1):
                if NBelow[Distance-1] >= 2 and NBelow[Distance-2] >= 2:
                    break
                StopIndex = Distance
            
            if LayersWithAtLeastTwoHitsBetweenStartAndStop < LayerRequirement:
                nRejectedClustering += 1
                continue

            # Search N layers after candidate for first layer with 2+ hits -- NEW
            selected_layer_hits = None
            selected_distance = None

            SearchLayersFor2Hit = 5 # CHANGE THIS FOR DESIRED NUMBER OF LAYERS BELOW CANDIDATE BEFORE 2+ HITS REQUIREMENT IS ENFORCED (new logic)

            for Distance in range(1, SearchLayersFor2Hit+1):
                layer_hits = [rese for rese in RESEs if self.Geometry.GetLayerDistance(candidate, rese) == -Distance] 

                if len(layer_hits) >= 2:
                    selected_layer_hits = layer_hits
                    selected_distance = Distance
                    break

            if selected_layer_hits is None:
                continue

            # -------------------------------
            # IF PERFORMING A POLARIZATION ANALYSIS

            if polarization == True:
                # Project along MC direction to selected layer
                init_pos = np.array([candidate.GetPosition().X(),
                    candidate.GetPosition().Y(),
                    candidate.GetPosition().Z()])

                # Move to the identified layer below candidate vertex and project virtual point on that layer
                #for i in self.DetectorList:
                    #print(Geometry.GetDetector(i).GetStructuralPitch().Z())
                # Detector = Geometry.GetDetector("AstroPix") # accessing the tracker strip AstroPix information - MAKE MORE UNIVERSAL? -> look into something like referencing the 2D Strip DetectorType (-1 I think)
                VF = VertexFinder(self.Geometry, NumberOfLayers=self.NumberOfLayers)
                interlayer_distance = VF.interlayer_distance 
                z_target = candidate.GetPosition().Z() - selected_distance * interlayer_distance

                init_dir = initial_vector_function(theta, phi) 
                projected_point = project_to_layer(init_pos, init_dir, z_target)
            else:
                init_dir = None
            # -------------------------------


            '''
            The parts below are selecting specific event morphologies and using their geometry to reconstruct
            the gamma-ray direction
            '''

            # ------------------------------
            # FOR EVENTS WHOSE FIRST HIT IS FOLLOWED BY A LAYER WITH 2+ HITS
            if selected_distance == 1:
        
                # Choose best two hits in that layer
                if init_dir is not None:
                    # if using MC direction information, take two closest hits to the projected point
                    hit1, hit2, filtered_hits = select_two_closest_hits(selected_layer_hits, projected_point)
                elif init_dir is None:
                    # if no MC information is used, cluster closest hits until two remain -> reconstruct using those
                    #clustered = clustering_hits_bydist(selected_layer_hits)
                    candidate_position = np.array([candidate.GetPosition().X(), candidate.GetPosition().Y(), candidate.GetPosition().Z()])
                    clustered = clustering_hits_by_angle(selected_layer_hits, candidate_position)
                    #clustered = cluster_by_distance(selected_layer_hits, 0.05)
                    if len(clustered) >= 2:
                        hit1, hit2 = select_best_pair(clustered, candidate_position)
                        #hit1, hit2 = select_best_pair(clustered, candidate_position)
                    else:
                        hit1, hit2 = None, None           
            # ------------------------------


            # ------------------------------
            # FOR EVENTS THAT HAVE MULTIPLE SINGLE HITS BEFORE TWO HITS
            # assign hit one AND hit two as the hit in the layer below single hit
            if selected_distance > 1:
                # Get hits in the layer immediately below the candidate
                layer_hits_below = [rese for rese in RESEs if self.Geometry.GetLayerDistance(candidate, rese) == -selected_distance] # this is throwing out events
                candidate_position = np.array([candidate.GetPosition().X(), candidate.GetPosition().Y(), candidate.GetPosition().Z()])
                if len(layer_hits_below) >= 2:
                    hit1, hit2 = layer_hits_below[0], layer_hits_below[1]
                else:
                    hit1 = None
                    hit2 = None
            # ------------------------------


            if hit1 is None or hit2 is None:
                nRejectedNo2HitLayer += 1
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
                if selected_distance == 1: # handles events with 1ht followed by 2ht (typical morphology for pair production events)
                    electron_track = np.array([hit1.GetPosition().X(),
                                            hit1.GetPosition().Y(),
                                            hit1.GetPosition().Z()]) \
                                    - np.array([vtx.x, vtx.y, vtx.z])
                    positron_track = np.array([hit2.GetPosition().X(),
                                            hit2.GetPosition().Y(),
                                            hit2.GetPosition().Z()]) \
                                    - np.array([vtx.x, vtx.y, vtx.z])
                    
                    vtx.vertex_type = 'type_1ht2ht' # assigning a type to the event for histogramming purposes 
                    vtx.electron_energy = hit1.GetEnergy()
                    vtx.positron_energy = hit2.GetEnergy()

                    electron_track /= np.linalg.norm(electron_track)
                    positron_track /= np.linalg.norm(positron_track)
                    vtx.electron_dir = electron_track / np.linalg.norm(electron_track)
                    vtx.positron_dir = positron_track / np.linalg.norm(positron_track)

                    gamma_dir = vtx.ComputeIncomingGammaDirection()
                    vtx.gamma_dir = gamma_dir
                    #print(f"Event {RE.GetEventID()}| type: {vtx.vertex_type} | vertex z: {vtx.GetZPosition()} | gamma dir: {gamma_dir}")
                
                # in order to reconstruct the gamma-ray for events that have single hits, reconstruct by drawing a line
                #        from the vertex to the single hit and treating that as both the electron and positron track
                # for 1ht-1ht events, use the first two recorded hits to define gamma-ray direction
                if selected_distance > 1 and len(RESEs) >= 2: # handles cases where there are multiple single hits before the two hits
                    first_hit = RESEs[0] # first hit in the event
                    second_hit = RESEs[1] # second hit in the event

                    first_hit_pos = np.array([first_hit.GetPosition().X(),
                            first_hit.GetPosition().Y(),
                            first_hit.GetPosition().Z()])
                    second_hit_pos = np.array([second_hit.GetPosition().X(),
                                            second_hit.GetPosition().Y(),
                                            second_hit.GetPosition().Z()])
                    
                    best_track = np.array([vtx.x, vtx.y, vtx.z]) - first_hit_pos
                    alternate_track = second_hit_pos - np.array([vtx.x, vtx.y, vtx.z])

                    if np.allclose(first_hit_pos, np.array([vtx.x, vtx.y, vtx.z])):
                        electron_track = alternate_track
                        positron_track = alternate_track
                    elif np.allclose(second_hit_pos, np.array([vtx.x, vtx.y, vtx.z])):
                        electron_track = best_track
                        positron_track = best_track
                    else:
                        electron_track = best_track
                        positron_track = best_track
                    
                    electron_track /= np.linalg.norm(electron_track)
                    positron_track /= np.linalg.norm(positron_track)
                    gr_direction = -(electron_track + positron_track)
                    gr_direction /= np.linalg.norm(gr_direction)
                    vtx.vertex_type = 'type_1ht1ht' 
                    vtx.electron_dir = electron_track
                    vtx.positron_dir = positron_track
                    vtx.electron_energy = first_hit.GetEnergy()
                    vtx.positron_energy = second_hit.GetEnergy()
                    gamma_dir = vtx.ComputeIncomingGammaDirection()
                    vtx.gamma_dir = gamma_dir

                    #print(f"Event {RE.GetEventID()}| type: {vtx.vertex_type} | vertex z: {vtx.GetZPosition()} | gamma dir: {gr_direction}")
                #print(f"Event {RE.GetEventID()}: Angle between: {np.arccos(np.clip(np.dot(gr_direction, init_dir), -1.0, 1.0)) * 180/np.pi:.2f} degrees, event type:{vertex_type}")
                true_dir = initial_vector_function(theta, phi) # THIS CAN ONLY BE DONE IF POLARIZATION FLAG SET TO TRUE

            
        '''
        SOME NEW LOGIC STARTS HERE
        '''
        # New logic for events that did not have a vertex assigned (2ht 2ht morphology)

        if not vertex_created_for_event:
            
            # Identify the hits in the topmost layer in the event
            top_hit = RESEs[0]
            hits_in_first_layer = [rese for rese in RESEs if self.Geometry.GetLayerDistance(rese, top_hit) == 0] # in same layer as the shallowest hit

            if len(hits_in_first_layer) == 2: # restricting to only allow exactly two hits in the top layer

                self.two_hit_after_dead_material_counter += 1 # ONLY FOR DEAD MATERIAL ANALYSIS

                hit1, hit2 = hits_in_first_layer

                # Now look for hits in the layer immediately below
                hits_in_layer_below = [rese for rese in RESEs if self.Geometry.GetLayerDistance(hit1, rese) == -1]

                # for now also I am also requiring exactly two hits in the layer below -- CAN CHANGE NOW USING CLUSTERING FOR SECOND LAYER HITS
                if len(hits_in_layer_below) == 2:
                    
                    # mimicking the logic applied to the 1ht events
                    NBelow = [0] * self.SearchRange
                    for rese in RESEs:
                        Distance = self.Geometry.GetLayerDistance(top_hit, rese)
                        if Distance < 0 and abs(Distance) < self.SearchRange:
                            NBelow[abs(Distance)] += 1

                    StartIndex = 0
                    StopIndex = 0
                    LayersWithAtLeastTwoHitsBetweenStartAndStop = 0

                    for Distance in range(1, self.SearchRange-1):
                        if NBelow[Distance] == 0 and NBelow[Distance+1] == 0:
                            break
                        StopIndex = Distance

                        if StartIndex == 0 and NBelow[Distance] > 1:
                            StartIndex = Distance

                        if StartIndex != 0 and NBelow[Distance] >= 2:
                            LayersWithAtLeastTwoHitsBetweenStartAndStop += 1

                    if LayersWithAtLeastTwoHitsBetweenStartAndStop < LayerRequirement:
                        return Vertices  # skip event

                    self.two_hit_in_two_layers_after_dead_material_counter += 1 # ONLY FOR DEAD MATERIAL ANALYSIS

                    hit1lay1, hit2lay1 = hits_in_first_layer
                    hit1lay2, hit2lay2 = hits_in_layer_below

                    # getting positions of all the hits (A is first layer, B is second layer)
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
                    (track1, track2) = self.BestHitPairing(A1, A2, B1, B2)

                    (p1, q1), (p2, q2) = track1, track2 # p = point in layer 1, q = point in layer 2

                    v1 = q1-p1
                    v2 = q2-p2

                    # Assigning the vertex a 3D position
                    vertex_point = self.CalculatingVertexPosition(p1, v1, p2, v2)
                    
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
        #print(f"Event {RE.GetEventID()}: event type: {vertex_type}")

        return Vertices

    def TopVertex(self, vertex_list):
        '''
        Select the top-most vertex (maximum z value) from a list of vertex candidates
        ---------------------------------------------------------------
        Parameters:
            vertex_list (list): List of Vertex objects

        Returns:
            Vertex: The vertex with the highest Z position
        '''

        return max(vertex_list, key=lambda v: v.GetZPosition())        


'''
(e) EVENTPLOTTING CLASS
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
        #Clusterizer.SetParameters(0,0,0,0,0,0,0,0,True) # Default clustering parameters are (1,-1)...CURRENT INPUT STOPS CLUSTERING
        Clusterizer.SetParameters(-1, 1)

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
    
    def PlotGammaRayReconstructionHistogram(self, all_angle_differences, oneht_twoht_angles=None, oneht_oneht_angles=None, twoht_twoht_angles=None, revan_angles=None): # , revan_angle_differencess
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        #Plot histogram of angle differences between reconstructed and true gamma-ray directions
        #--------------------------------------------------------------
        #Parameters:
        #    angle_differences (list): List of angle differences (between reconstructed and actual) in degrees
        #    inputfile (str): filename
        
        #Returns:
        #    None (displays a histogram plot)
        

        all_values = []
        if oneht_twoht_angles:
            all_values += oneht_twoht_angles
        if oneht_oneht_angles:
            all_values += oneht_oneht_angles
        if twoht_twoht_angles:
            all_values += twoht_twoht_angles

        binwidth = 3
        bins = np.arange(min(all_values), max(all_values) + binwidth, binwidth)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        plt.figure(figsize=(4.5,3))
        # 1 hit 2 hit
        if oneht_twoht_angles:
            counts, _ = np.histogram(oneht_twoht_angles, bins=bins)
            plt.plot(bin_centers, counts, color='orange', alpha=0.7, linewidth=2, linestyle='dashdot', label=f'1 hit 2 hit ({len(oneht_twoht_angles)} events)')

        # 1 hit 1 hit
        if oneht_oneht_angles:
            counts, _ = np.histogram(oneht_oneht_angles, bins=bins)
            plt.plot(bin_centers, counts, color='forestgreen', alpha=0.7, linewidth=2, linestyle='dashed', label=f'1 hit 1 hit ({len(oneht_oneht_angles)} events)')

        # 2 hit 2 hit
        if twoht_twoht_angles:
            counts, _ = np.histogram(twoht_twoht_angles, bins=bins)
            plt.plot(bin_centers, counts, color='red', alpha=0.7, linewidth=2, linestyle='dotted', label=f'2 hit 2 hit ({len(twoht_twoht_angles)} events)')

        if all_values:
            counts, _ = np.histogram(all_values, bins=bins)
            plt.plot(bin_centers, counts, color='black', alpha=0.7, linewidth=2, label=f'New ({len(all_values)} total events)')
        # Old method (Revan)
        if revan_angles:
            counts, _ = np.histogram(revan_angles, bins=bins)
            plt.plot(bin_centers, counts, color='darkcyan', alpha=0.7, linewidth=2, zorder=3, label=f'Old ({len(revan_angles)} total events)')

        plt.xlabel(r"Angle between reconstructed and true direction [${}^\circ$]", fontsize=12)
        plt.ylabel("Number of events", fontsize=12)
        plt.title("Gamma-ray Reconstruction Accuracy", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.minorticks_on()
        #plt.text(0,1, f'New method total events: {len(all_values)}', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=1))
        plt.savefig('reconstruction_histogram.png', transparent=True, bbox_inches="tight", dpi=400)
        plt.show()

    '''
    def PlotGammaRayReconstructionHistogram(self, all_angle_differences, oneht_twoht_angles=None, oneht_oneht_angles=None, twoht_twoht_angles=None, revan_angles=None): # , revan_angle_differencess
    
        #Plot histogram of angle differences between reconstructed and true gamma-ray directions
        #--------------------------------------------------------------
        #Parameters:
        #    angle_differences (list): List of angle differences (between reconstructed and actual) in degrees
        #    inputfile (str): filename
        
        #Returns:
        #    None (displays a histogram plot)
        

        all_values = []
        if oneht_twoht_angles:
            all_values += oneht_twoht_angles
        if oneht_oneht_angles:
            all_values += oneht_oneht_angles
        if twoht_twoht_angles:
            all_values += twoht_twoht_angles

        binwidth = 1
        bins = np.arange(min(all_values), max(all_values) + binwidth, binwidth)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        # Plot each histogram
        # plt.hist(all_angle_differences, bins=bins, alpha=1, linewidth=2, edgecolor='black', cumulative=True, density=True, histtype='step', label='All reconstructed events (current logic)')

        if oneht_twoht_angles:
            n_1ht2ht, bins, patches = plt.hist(oneht_twoht_angles, bins=bins, alpha=0.7, linewidth=2, edgecolor='blue', cumulative=True, density=True, histtype='step', label=f'1 hit 2 hit ({len(oneht_twoht_angles)} events)')
            #plt.plot(bin_centers, n_1ht2ht, color='blue', alpha=0.6, linewidth=2)

        if oneht_oneht_angles:
            n_1ht1ht, bins, patches = plt.hist(oneht_oneht_angles, bins=bins, alpha=0.7, linewidth=2, edgecolor='orange', cumulative=True, density=True, histtype='step', label=f'1 hit 1 hit ({len(oneht_oneht_angles)} events)')
            #plt.plot(bin_centers, n_1ht1ht, color='orange', alpha=0.6, linewidth=2)

        if twoht_twoht_angles:
            n_2ht2ht, bins, patches = plt.hist(twoht_twoht_angles, bins=bins, alpha=0.7, linewidth=2, edgecolor='red', cumulative=True, density=True, histtype='step', label=f'2 hit 2 hit ({len(twoht_twoht_angles)} events)')
            #plt.plot(bin_centers, n_2ht2ht, color='red', alpha=0.6, linewidth=2)

        if revan_angles:
            n_revan, bins, patches = plt.hist(revan_angles, bins=bins, alpha=0.7, linewidth=2, edgecolor='green', cumulative=True, density=True, histtype='step', label=f'Old method ({len(revan_angles)} total events)', linestyle = 'dashed', zorder = 3)
            #plt.plot(bin_centers, n_revan, color='green', alpha=0.6, linewidth=2)

        plt.xlabel(r"Difference between reconstructed and true $\gamma$-ray direction [${}^\circ$]", fontsize=12)
        plt.ylabel("Normalized number of events", fontsize=12)
        plt.title("Gamma-Ray Reconstruction Accuracy", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.minorticks_on()
        plt.text(0,1, f'New method total events: {len(all_values)}', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=1))
        plt.show()
    '''
    '''
    # USE THE FOLLOWING FOR TOTAL LINE AND SEPARATED HT BY HT RECONSTRUCTION
    def PlotGammaRayReconstructionHistogram(
        self,
        all_angle_differences,
        oneht_twoht_angles=None,
        oneht_oneht_angles=None,
        twoht_twoht_angles=None,
        revan_angles=None
    ):

        import numpy as np
        import matplotlib.pyplot as plt

        # -----------------------------
        # Combine all event types
        # -----------------------------
        all_values = []

        if oneht_twoht_angles:
            all_values += oneht_twoht_angles

        if oneht_oneht_angles:
            all_values += oneht_oneht_angles

        if twoht_twoht_angles:
            all_values += twoht_twoht_angles

        total_events = len(all_values)

        # -----------------------------
        # Common binning
        # -----------------------------
        binwidth = 1
        bins = np.arange(min(all_values), max(all_values) + binwidth, binwidth)

        # -----------------------------
        # TOTAL histogram
        # -----------------------------
        total_weights = np.ones(len(all_values)) / total_events

        plt.hist(
            all_values,
            bins=bins,
            weights=total_weights,
            histtype='step',
            linewidth=2,
            color='black',
            cumulative=True,
            label=f'New ({total_events} total events)',
            zorder=5
        )

        # -----------------------------
        # 1 hit 2 hit
        # -----------------------------
        if oneht_twoht_angles:

            weights_1ht2ht = np.ones(len(oneht_twoht_angles)) / total_events

            plt.hist(
                oneht_twoht_angles,
                bins=bins,
                weights=weights_1ht2ht,
                histtype='step',
                linewidth=2,
                color='blue',
                cumulative=True,
                label=f'1 hit 2 hit ({len(oneht_twoht_angles)} events)'
            )

        # -----------------------------
        # 1 hit 1 hit
        # -----------------------------
        if oneht_oneht_angles:

            weights_1ht1ht = np.ones(len(oneht_oneht_angles)) / total_events

            plt.hist(
                oneht_oneht_angles,
                bins=bins,
                weights=weights_1ht1ht,
                histtype='step',
                linewidth=2,
                color='orange',
                cumulative=True,
                label=f'1 hit 1 hit ({len(oneht_oneht_angles)} events)'
            )

        # -----------------------------
        # 2 hit 2 hit
        # -----------------------------
        if twoht_twoht_angles:

            weights_2ht2ht = np.ones(len(twoht_twoht_angles)) / total_events

            plt.hist(
                twoht_twoht_angles,
                bins=bins,
                weights=weights_2ht2ht,
                histtype='step',
                linewidth=2,
                color='red',
                cumulative=True,
                label=f'2 hit 2 hit ({len(twoht_twoht_angles)} events)'
            )

        # -----------------------------
        # Revan comparison
        # -----------------------------
        if revan_angles:

            weights_revan = np.ones(len(revan_angles)) / len(revan_angles)

            plt.hist(
                revan_angles,
                bins=bins,
                weights=weights_revan,
                histtype='step',
                linewidth=2,
                color='green',
                linestyle='dashed',
                cumulative=True,
                label=f'Old ({len(revan_angles)} total events)',
                zorder=3
            )

        plt.xlabel(r"Difference between reconstructed and true $\gamma$-ray direction [${}^\circ$]")
        plt.ylabel("Cumulative fraction of events")
        plt.title("Gamma-Ray Reconstruction Accuracy")

        plt.grid(linestyle='--')
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()

        plt.show()
    '''
    

    def GammaRayReconstructionScatterplot(self, all_angle_differences, revan_angles):
        '''
        Scatter plot of angle differences between reconstructed and true gamma-ray directions for current logic vs. Revan
        --------------------------------------------------------------
        Parameters:
            all_angle_differences (list): List of angle differences (between reconstructed and actual) in degrees for current logic
            revan_angles (list): List of angle differences for Revan reconstruction
        Returns:
            None (displays a scatter plot)
        '''
        plt.figure(figsize=(8, 6))
        plt.scatter(all_angle_differences, revan_angles, color='blue', alpha=0.6)
        plt.xlabel('Updated Reconstruction')
        plt.ylabel("Revan Reconstruction")
        plt.title("Gamma-Ray Reconstruction Accuracy: Current Logic vs. Revan")
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
(f) MCINTERACTION CLASS
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
    parser.add_argument('--revan', action='store_true', help='Include Revan reconstructed events in the gamma-ray reconstruction histogram (requires Revan reconstruction to have already been run on the same input file)')
    parser.add_argument('--gr-scatterplot', action='store_true', help='Plot scatterplot of gamma-ray reconstruction angle difference for new reconstruction logic vs. Revan reconstruction')
    parser.add_argument('--polarization', action='store_true', default=False, help='Pass this flag if doing polarization analysis, omit if not')
    parser.add_argument('--o', type=str, help='Enter the path for the output file for angle difference text file')
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
    outputpath = args.o

    # Read in events (note that MFileEventsEvta automatically applies noising -- essential for realistic results)
    Reader = M.MFileEventsEvta(Geometry)
    Reader.Open(M.MString(inputfile))

    # Identify vertices in the input data
    VF = VertexFinder(Geometry, NumberOfLayers=NumberOfLayers)

    # Cluster the events
    Clusterizer = M.MERHitClusterizer()
    Clusterizer.SetGeometry(Geometry)
    Clusterizer.SetParameters(0,0,0,0,0,0,0,0,True) # turns clustering off
    #Clusterizer.SetParameters(-1, 1) -> turns default clustering on
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
    output_phi_filename = f"{inputfile}_60degCUTOFFPhiValues.txt"

    # Remove output file if it already exists
    if os.path.exists(output_phi_filename):
        os.remove(output_phi_filename)

    EP = EventPlotting(inputfile, Geometry)
    HTX, HTY, HTZ, ElectronHTX, ElectronHTY, ElectronHTZ, PositronHTX, PositronHTY, PositronHTZ = EP.GetSimHits()

    all_vertices = []

    # Setting up file names
    if inputfile.endswith('.sim.gz'):
        base_filename = inputfile[:-7]
        filetype  = '.sim.gz'
    elif inputfile.endswith('.sim'):
        base_filename = inputfile[:-4]
        filetype = '.sim'
    else:
        raise ValueError('Input file must be of type .sim or .sim.gz')
    # Reading each event and stopping if none left, rejecting "invalid" events (as identified in MEGAlib)

    angle_output_file = open(f"{outputpath}{inputfile}_NewAngleDiffs.txt", "w")    

    while True:
        RE = Reader.GetNextEvent()
        M.SetOwnership(RE, True) # necessary to avoid memory leaks

        if not RE:
            print("No more events.")
            break
        if not RE.IsValid():
            #print(f"Skipping invalid event at count {EventCount}.") # this seems to correspond to an energy deposit of 0 keV
            continue

        EventNumberToEventID[EventCount] = RE.GetEventID()
        REI = M.MRawEventIncarnations()
        M.SetOwnership(REI, True)

        REI.SetInitialRawEvent(RE)
        Clusterizer.Analyze(REI)
        ClusteredRE = REI.GetInitialRawEvent()
        M.SetOwnership(ClusteredRE, True)

        #Detector = Geometry.GetDetector("AstroPix") # accessing the tracker strip AstroPix information - MAKE MORE UNIVERSAL? -> look into something like 
        VF = VertexFinder(Geometry, NumberOfLayers=NumberOfLayers)
        interlayer_distance = VF.interlayer_distance

        # Getting the vertices and assigning each to its corresponding event ID -> VertexDict
        Vertices = VF.FindVertices(ClusteredRE, theta=args.theta, phi=args.phi, polarization=args.polarization)
        VertexDict[RE.GetEventID()] = Vertices 

        if Vertices:
            top_vertex = VF.TopVertex(Vertices)
            vertex_z = top_vertex.GetPosition()[2]
            if top_vertex.gamma_dir is None:
                print(f"WARNING: EID {RE.GetEventID()} has None gamma_dir")
                NumberOfVerticesPerEvent.append(len(Vertices))
                if len(Vertices) > 0:
                    PairsFound += 1
                EventCount += 1
                continue

            # Reconstructed off-axis angle: angle between the reconstructed gamma-ray direction and the instrument's z-axis
            u_rec = top_vertex.gamma_dir / np.linalg.norm(top_vertex.gamma_dir)
            reco_offaxis_angle = np.degrees(np.arccos(np.clip(u_rec[2], -1.0, 1.0)))

            # Reject events with a reconstructed off-axis angle greater than X degrees,
            # before the vertex is recorded anywhere (all_vertices, phi output, etc.)
            if reco_offaxis_angle > 60:
                NumberOfVerticesPerEvent.append(len(Vertices))
                if len(Vertices) > 0:
                    PairsFound += 1
                EventCount += 1
                continue

            all_vertices.append(top_vertex)

            theta_rad = np.radians(args.theta)
            phi_rad = np.radians(args.phi)

            u_mc = np.array([
                np.sin(theta_rad) * np.cos(phi_rad),
                np.sin(theta_rad) * np.sin(phi_rad),
                np.cos(theta_rad)
            ])
            u_mc = u_mc / np.linalg.norm(u_mc)

            cos_angle = np.clip(np.dot(u_rec, u_mc), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle is not None:
                angle_output_file.write(f"{RE.GetEventID()} {angle}\n")

            print(f"EID: {RE.GetEventID()} "
                  f"| Final event vertex position: ({top_vertex.GetPosition()[0]:.6f}, {top_vertex.GetPosition()[1]:.6f}, {top_vertex.GetPosition()[2]:.6f}) "
                  f"| event type: {top_vertex.vertex_type} "
                  f"| gamma-ray direction: ({top_vertex.gamma_dir[0]:.6f}, {top_vertex.gamma_dir[1]:.6f}, {top_vertex.gamma_dir[2]:.6f})")
            
            hits_below = get_hits_in_layer_below(RE.GetEventID(), vertex_z, HTX, HTY, HTZ, interlayerdistance=interlayer_distance)

            # Projected point -> make function
            init_dir = initial_vector_function(args.theta, args.phi)
            
            initial_pos = np.array([
                top_vertex.GetPosition()[0],
                top_vertex.GetPosition()[1],
                vertex_z
            ])

            z_target = vertex_z - interlayer_distance # cm (interlayer distance)
            projected_point = project_to_layer(initial_pos, init_dir, z_target)

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
            
            #phi = top_vertex.ComputePhi(theta = args.theta, phi = args.phi, hit1=top_vertex.AllRESEs[0], hit2=top_vertex.AllRESEs[1], ref_direction = args.ref_dir)
            phi = top_vertex.ComputePhi_RelativeX(photon_dir = top_vertex.gamma_dir, hit1=top_vertex.AllRESEs[0], hit2=top_vertex.AllRESEs[1])
            if phi is not None:
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
            #print(f"Processed {EventCount} events...", flush=True)

    # Compute true direction
    angle_output_file.close()

    true_dir = initial_vector_function(args.theta, args.phi)

    all_angle_differences = []
    oneht_twoht_angles = []
    oneht_oneht_angles = []
    twoht_twoht_angles = []

    for vtx in all_vertices:
        
        if vtx.gamma_dir is None:
            #print(f"WARNING: vertex missing gamma_dir | type: {getattr(vtx, 'vertex_type', 'unknown')}")
            continue
        
        cos_angle = np.clip(np.dot(vtx.gamma_dir, true_dir), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        all_angle_differences.append(angle_deg)

        if hasattr(vtx, "vertex_type"):
            if vtx.vertex_type == "type_1ht2ht":
                oneht_twoht_angles.append(angle_deg)
            elif vtx.vertex_type == "type_1ht1ht":
                oneht_oneht_angles.append(angle_deg)
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
        if args.revan is True:
            revan_angle_differences = np.loadtxt(f"{base_filename}_revan_angle_differences.txt") # load in the Revan angle differences from the text file
            revan_angle = revan_angle_differences.tolist() # convert to list for histogramming
            EP.PlotGammaRayReconstructionHistogram(all_angle_differences, oneht_twoht_angles, oneht_oneht_angles, twoht_twoht_angles, revan_angles=revan_angle) # histogramming with revan info
        else:
            #EP.PlotGammaRayReconstructionHistogram(all_angle_differences, oneht_twoht_angles, oneht_oneht_angles, twoht_twoht_angles) # histogramming without revan info
            print("not plotting histogram")

        print("\n--------- OVERALL RECONSTRUCTION PERFORMANCE ---------")
        print("Total # of reconstructed events:", len(all_vertices))
        print("Average reconstructed direction:", mean_dir)
        print("True direction (from MC input):", true_dir)
        print("Median angle between reconstructed and MC truth:", f"{np.median(all_angle_differences):.3f} degrees")

        # PERCENTILE CALCULATIONS
        if len(all_angle_differences) > 0:
            allpercentile68 = np.percentile(all_angle_differences, 68) 
            print(f"68th percentile angle difference for all reconstructed events: {allpercentile68:.3f} degrees")
            np.save(f"{inputfile}_all_angle_differences.npy", np.array(all_angle_differences))
        else:
            allpercentile68 = None
            print("No reconstructed events found; cannot compute 68th percentile.")
        # -----------------------------
        if len(oneht_twoht_angles) > 0:
            oneht2htpercentile68 = np.percentile(oneht_twoht_angles, 68) 
            print(f"68th percentile angle difference for 1 hit 2 hit events: {oneht2htpercentile68:.3f} degrees, {len(oneht_twoht_angles)} events")
            np.save(f"{inputfile}_oneht_twoht_angles.npy", np.array(oneht_twoht_angles))
        else:
            oneht2htpercentile68 = None
            print("No 1 hit 2 hit events found; cannot compute 68th percentile.")
        # -----------------------------
        if len(oneht_oneht_angles) > 0:
            oneht1htpercentile68 = np.percentile(oneht_oneht_angles, 68)
            print(f"68th percentile angle difference for 1 hit 1 hit events: {oneht1htpercentile68:.3f} degrees, {len(oneht_oneht_angles)} events")
            np.save(f"{inputfile}_oneht_oneht_angles.npy", np.array(oneht_oneht_angles))
        else:
            oneht1htpercentile68 = None
            print("No 1 hit 1 hit events found; cannot compute 68th percentile.")
        # -----------------------------
        if len(twoht_twoht_angles) > 0:
            twoht2htpercentile68 = np.percentile(twoht_twoht_angles, 68)
            print(f"68th percentile angle difference for 2 hit 2 hit events: {twoht2htpercentile68:.3f} degrees, {len(twoht_twoht_angles)} events")
            np.save(f"{inputfile}_twoht_twoht_angles.npy", np.array(twoht_twoht_angles))
        else:
            twoht2htpercentile68 = None
            print("No 2 hit 2 hit events found; cannot compute 68th percentile.")
        # -----------------------------
        if args.revan is True and len(revan_angle_differences) > 0:
            revanpercentile68 = np.percentile(revan_angle_differences, 68)
            print(f"68th percentile angle difference for Revan reconstructed events: {revanpercentile68:.3f} degrees")
        elif args.revan is True:
            print("No Revan reconstructed events found in provided file. Cannot compute 68th percentile.")
    
    if args.gr_scatterplot:
        if args.revan is True:
            revan_angle_differences = np.loadtxt(f"{base_filename}_revan_angle_differences.txt") # load in the Revan angle differences from the text file
            revan_angle = revan_angle_differences.tolist() # convert to list for scatterplotting
            EP.GammaRayReconstructionScatterplot(all_angle_differences, revan_angles=revan_angle) # scatterplotting with revan info
        else:
            print("Revan reconstruction data not provided; cannot generate scatterplot comparing to Revan.")
    
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