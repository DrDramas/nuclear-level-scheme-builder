#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday April 5, 2023
Last updated on Wednesday, April 22, 2026

@author: tbudner, dlenz, mcarpenter
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from functools import cached_property

import numpy as np
import numpy.random as rnd
import pandas as pd
import networkx as nx
from textwrap import dedent
import copy

# ###################################################### # 
class LSGraph(object):
    """Custom graph implementation for nuclear level schemes.

    Represents the decay scheme as an adjacency-list graph where each
    vertex is a nuclear level energy (float) and edges connect levels
    that are linked by a gamma-ray transition.

    The class provides path-finding utilities used to enumerate all
    possible gamma-ray cascades from an excited state to the ground state.

    Parameters
    ----------
    graph_dict:
        Initial adjacency dictionary ``{energy: [neighbour_energies]}``.
        Defaults to an empty graph.
    """

    def __init__(self, graph_dict=None):
        """Initialize a graph object.

        Parameters
        ----------
        graph_dict:
            Adjacency dictionary {vertex: [neighbours]}. If None or
            omitted, an empty graph is created.
        """
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def edges(self, vertice):
        #""" returns a list of all the edges of a vertice"""
        return self._graph_dict[vertice]
        
    def all_vertices(self):
        #""" returns the vertices of a graph as a set """
        return set(self._graph_dict.keys())

    def all_edges(self):
        #""" returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        #""" If the vertex is not in 
        #    self._graph_dict, a key vertex with an empty
        #    list as a value is added to the dictionary. 
        #    Otherwise nothing has to be done. 
        #"""
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        #""" assumes that edge is of type set, tuple or list; 
        #    between two vertices can be multiple edges! 
        #"""
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self._graph_dict:
                self._graph_dict[x].add(y)
            else:
                self._graph_dict[x] = [y]

    def __generate_edges(self):
        """Generate all unique edges of the graph.

        Returns
        -------
        list[frozenset]
            Each element is a frozenset of one or two vertices
            representing an edge.
        """
        seen: set[frozenset] = set()
        edges: list[frozenset] = []
        for vertex in self._graph_dict:
            for neighbour in self._graph_dict[vertex]:
                fs = frozenset({vertex, neighbour})
                if fs not in seen:
                    seen.add(fs)
                    edges.append(fs)
        return edges
    
    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj
    
    def __next__(self):
        #""" allows us to iterate over the vertices """
        return next(self._iter_obj)

    def __str__(self):
        res = "vertices: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res
    
    def find_path(self, start_vertex, end_vertex, path=None):
        """Find a path from start_vertex to end_vertex in the graph.

        Parameters
        ----------
        start_vertex:
            Starting node.
        end_vertex:
            Target node.
        path:
            Accumulated path so far (do not pass — used internally).

        Returns
        -------
        list or None
            First path found, or None if no path exists.
        """
        if path is None:
            path = []
        graph = self._graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path:
                    return extended_path
        return None


    def find_all_paths(self, start_vertex, end_vertex, path=None):
        """Find all paths from start_vertex to end_vertex in the graph.

        Parameters
        ----------
        start_vertex:
            Starting node.
        end_vertex:
            Target node.
        path:
            Accumulated path so far (do not pass — used internally).

        Returns
        -------
        list[list]
            All simple paths between the two vertices.
        """
        if path is None:
            path = []
        graph = self._graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths:
                    paths.append(p)
        return paths

# ###################################################### # 
class Gamma:
    """Represents a single gamma-ray transition between two nuclear levels.

    Parses a fixed-width formatted line from an IAEA .gam data file and
    stores the transition's physical properties.

    Parameters
    ----------
    line : str
        A single line from a .gam file.

    Attributes
    ----------
    gE : float
        Gamma-ray energy in keV.
    gE_err : int
        Uncertainty on gE (in last digits).
    RI : float
        Relative decay intensity.
    RI_err : int
        Uncertainty on RI.
    Multline : str
        Multipolarity string (e.g. "E1", "M1+E2").
    CC : float
        Internal conversion coefficient.
    ExT : float
        Excitation energy of the initial (top) nuclear level in keV.
    SpnT : str
        Spin-parity of the initial level.
    ExB : float
        Excitation energy of the final (bottom) nuclear level in keV.
    SpnB : str
        Spin-parity of the final level.
    BR : float
        Branching ratio — fraction of decays from ExT that emit this gamma.
        Set by :meth:`Level.compute_BRs`, not by the constructor.
    isGam : bool
        True if the line was successfully parsed as a gamma transition.
    """

    def __init__(self, *args):

        line = args[0]

        # first decode the gamma energy
    
        geline = line[0:16]
        tmp = geline.split()
        self.isGam = True
        try:
            self.gE = float(tmp[0])
        except:
            self.gE = 0.0
            self.isGam = False
        self.gE_err = 0
        if len(tmp) == 2:
            try:
                self.gE_err = int(tmp[1])
            except:
                self.gE_err = 0

        # decode relative intensity
        RIline = line[17:32]
        tmp = RIline.split()
        try:
            self.RI = float(tmp[0])
        except:
            self.RI = -1.0
        self.RI_err = 0
        if len(tmp) == 2:
            try:
                self.RI_err = int(tmp[1])
            except:
                self.RI_err = 0
        
        # Multipolarity info - not sure how to decode this yet
    
        self.Multline = line[33:59].strip()
    
        # conversion coefficents
        CCline = line[60:71]
        tmp = CCline.split()
        try:
            self.CC = float(tmp[0])
        except:
            self.CC = 0.0
        self.CC_err = 0
        if len(tmp) == 2:
            try:
                self.CC_err = int(tmp[1])
            except:
                self.C_err =0
    
        # Initial Excited level
        ExTline = line[72:87]
        tmp = ExTline.split()
        try:
            self.ExT = float(tmp[0])
        except:
            self.ExT = 0.0
        self.SpnT = "none"
        if len(tmp) == 2:
            try:
                self.SpnT = tmp[1].strip()
            except:
                self.SpnT = "none"
    
        # Final Excited level
        ExBline = line[88:104]
        tmp = ExBline.split()
        try:
            self.ExB = float(tmp[0])
        except:
            self.ExB = 0.0
        self.SpnB = "none"
        if len(tmp) == 2:
            try:
                self.SpnB = tmp[1].strip()
            except:
                self.SpnB = "none"
        
        # Branching ratio (to be determined for all gammas at each level)
        self.BR = 0
        
    def list_values(self) -> str:
        """Return a formatted string summarising this gamma-ray transition.

        The string is also printed to stdout for convenience.

        Returns
        -------
        str
            Multi-line summary of the transition's properties.
        """
        info = (
            f"Gamma Ray Transition Info\n"
            f"  Energy,      error : {self.gE} +/- {self.gE_err}\n"
            f"  Relative Int, error: {self.RI} +/- {self.RI_err}\n"
            f"  Multipolarity      : {self.Multline}\n"
            f"  Conv. Coeff, error : {self.CC} +/- {self.CC_err}\n"
            f"  Initial level (ExT): {self.ExT}  spin={self.SpnT}\n"
            f"  Final level   (ExB): {self.ExB}  spin={self.SpnB}\n"
        )
        print(info)
        return info

    def __repr__(self) -> str:
        return f"Gamma(E={self.gE} keV, RI={self.RI}, ExT={self.ExT}, ExB={self.ExB})"

# ###################################################### #
class Level:
    """Represents a single nuclear excited state (energy level).

    Stores all gamma-ray transitions that either populate (incoming)
    or de-excite (outgoing) this level.

    Parameters
    ----------
    ExE : float
        Excitation energy of this level in keV.

    Attributes
    ----------
    ExE : float
        Excitation energy in keV.
    out_gammas : list[Gamma]
        Gamma-ray transitions emitted from this level.
    in_gammas : list[Gamma]
        Gamma-ray transitions that populate this level.
    """

    def __init__(self, *args):

        self.ExE = args[0]
        self.out_gammas = []   # List of gamma rays emitted from this Level
        self.in_gammas=[] # List of transitions that directly populate this state
        
    def update_excitation_energy(self,exEn):
    
        self.ExE = exEn

    def add_outgoing_gamma(self,gamma):

        self.out_gammas.append(gamma)
        #print(self.ExE,gamma.gE)
    
    def add_incoming_gamma(self,gamma):

        self.in_gammas.append(gamma)
        #print(self.ExE,gamma.gE)

    def compute_BRs(self):

        total = 0
        for gamma in self.out_gammas:
            total += gamma.RI
        for gamma in self.out_gammas:
            gamma.BR = gamma.RI/total

    def list_gammas(self):

        for gamma in self.out_gammas:
            print("Initial Level:",self.ExE,"Gamma: ",gamma.gE,"Branching Ratio: ",gamma.BR)
            
# ###################################################### #
class LevelScheme:
    
    def __init__(self):
        
        self.nl = 0
        self.nt = 0
        self.g = nx.DiGraph()    # Directed graph object that represents the LevelScheme
        self.emax = 10000        # This is set for drawing
        self.levels = {}         # Dictionary {Level object : Node #} for LevelScheme
        self.placed = []         # List of gamma-rays that have already been placed in LevelScheme
        self.dummy_nodes = [] # List of nodes that have been placed unnecessarily and have been labeled redundant
        self.src_to_dst = {}       # Dictionary with the form {sourceNode:[destination1,destination2,destination3,...]}
        self.cascade = {}        # Dict {(source,destination):[pathway1,pathway2,pathway3,...]}; pathways are lists of gammas
        self.path_lengths = {}    # Dict {(source,destination):[length1,length2,length3,...]}; lengths are summed gamma energies
        self.savedNodes = []
        self.dummy_edges = []
        self.leaf_nodes = []
        self.root_nodes = []
        
    def __str__(self):

        return dedent(f"""
            Level Scheme with {self.nl} levels and {self.nt} transitions.
            Energies: {self.get_energies()}""")

    def get_energies(self):

        return [en for i, en in self.g.nodes.data('energy')]

    # def to_transition_scheme(self):
    #     return TransitionScheme.from_level_scheme(self)
    
    def add_level(self,lvl):
        
        nl=self.g.number_of_nodes()
        self.g.add_node(nl,energy=lvl.ExE)
        self.levels.update({nl:lvl})
    
    def map_from_transition_space(self,TS):
        
        for edge in TS.g.edges():
            # Scenario #1
            if edge[0] not in self.placed and edge[1] not in self.placed: # Make three new levels
                #print('-----Scenario 1------')
                # Add lowest level
                newLvl=Level(0.1)
                #newLvl.in_gammas.append(edge[1])
                newLvl.in_gammas.append(TS.g.nodes[edge[1]]['Gamma'])
                self.add_level(newLvl)
                # Add middle level
                newLvl=Level(0.1)
                #newLvl.in_gammas.append(edge[0])
                newLvl.in_gammas.append(TS.g.nodes[edge[0]]['Gamma'])
                #newLvl.out_gammas.append(edge[1])
                newLvl.out_gammas.append(TS.g.nodes[edge[1]]['Gamma'])
                self.add_level(newLvl)
                # Add highest level
                newLvl=Level(0.1)
                #newLvl.out_gammas.append(edge[0])
                newLvl.out_gammas.append(TS.g.nodes[edge[0]]['Gamma'])
                self.add_level(newLvl)
                # Add both transitions' energies to list of gammas placed in level scheme
                self.placed.append(edge[0])
                self.placed.append(edge[1])
            # Scenario #2
            elif edge[0] in self.placed and edge[1] not in self.placed:
                #print('-----Scenario 2------')
                for lvl in self.levels:
                    if TS.g.nodes[edge[0]]['Gamma'] in self.levels[lvl].in_gammas:
                        #lvl.out_gammas.append(edge[1]) # Add to existing list of outgoing gamma rays
                        self.levels[lvl].out_gammas.append(TS.g.nodes[edge[1]]['Gamma'])
                        newLvl=Level(0.1) # Add new level to LevelScheme
                        #newLvl.in_gammas.append(edge[1]) # Add previously unseen transition to list of incoming gamma rays
                        newLvl.in_gammas.append(TS.g.nodes[edge[1]]['Gamma'])
                        self.add_level(newLvl)
                        self.placed.append(edge[1]) # Add transition energy to list of placed gamma-rays
                        break
                        
            # Scenario #3
            elif edge[1] in self.placed and edge[0] not in self.placed:
                #print('-----Scenario 3------')
                for lvl in self.levels:
                    if TS.g.nodes[edge[1]]['Gamma'] in self.levels[lvl].out_gammas:
                        #lvl.in_gammas.append(edge[0]) # Add to existing list of incoming gamma rays
                        self.levels[lvl].in_gammas.append(TS.g.nodes[edge[0]]['Gamma'])
                        newLvl=Level(0.1) # Add new level to LevelScheme
                        #newLvl.out_gammas.append(edge[0]) # Add previously unseen transition to list of outgoing gamma rays
                        newLvl.out_gammas.append(TS.g.nodes[edge[0]]['Gamma'])
                        self.add_level(newLvl)
                        self.placed.append(edge[0]) # Add transition to list of placed gamma-rays
                        break
                        
            # Scenario #4
            else: # i.e. both gamma rays (edge[0] and edge[1]) have already been placed in the level scheme
                #print('-----Scenario 4------')
                #print('Gammas placed already: ',edge[0],edge[1])
                for lvl in self.levels:
                    #if edge[0] in lvl.in_gammas: # Find level which the edge[0] gamma populates
                    if TS.g.nodes[edge[0]]['Gamma'] in self.levels[lvl].in_gammas:
                        e0=lvl
                        #print('Level ',e0,' has incoming gamma ',edge[0])
                        break
                        
                for lvl in self.levels:
                    #if edge[1] in lvl.out_gammas: # Find level that emits the edge[1] gamma
                    if TS.g.nodes[edge[1]]['Gamma'] in self.levels[lvl].out_gammas:
                        e1=lvl
                        #print('Level ',e1,' has outgoing gamma ',edge[1])
                        break
                        
        #print('e0 level: ',e0)
        #print('e1 level: ',e1)
                if e0==e1:  # If these levels are the same, good. Move on to the next edge
                    continue
                else:  # Levels have been improperly duplicated. Remove the one more recently placed
                    for gamma in self.levels[e1].in_gammas: # Copy incoming gammes for redundant node/level
                        if gamma not in self.levels[e0].in_gammas:
                            self.levels[e0].in_gammas.append(gamma) # Add them to existing node/level
                    self.levels[e1].in_gammas.clear() # Delete this list so we don't accidentally reference redundant node/level
                    for gamma in self.levels[e1].out_gammas: # Copy outgoing gammes from redundant node/level
                        if gamma not in self.levels[e0].out_gammas: 
                            self.levels[e0].out_gammas.append(gamma) # Add them to existing node/level
                    self.levels[e1].out_gammas.clear() # Delete this list so we don't accidentally reference redundant node/level
                    if e1 not in self.dummy_nodes:
                        self.dummy_nodes.append(e1) # Add to list of redundant nodes
                        
    
    def connect_nodes_with_edges(self):
        
        for loLvl in self.levels:
            for gamma in self.levels[loLvl].in_gammas:
                for hiLvl in self.levels:
                    if gamma in self.levels[hiLvl].out_gammas:
                        self.g.add_edge(hiLvl,loLvl,energy=gamma.gE,weight=gamma.BR)
        
    def delete_edges_and_nodes(self):
        
        for edge in self.dummy_edges:
            self.g.remove_edge(edge[0],edge[1]) # Delete edge from graph
        self.dummy_edges.clear() # Once redundancies have been remove, empty the list of edges/nodes
        for node in self.dummy_nodes:
            self.g.remove_node(node) # Delete node from graph
            self.levels.pop(node) # Remove this Level from dictionaroy
        self.dummy_nodes.clear()
            
    def build_gamma_cascades(self,TS):
    
        for path in TS.all_paths:                 # Loop over all possible gamma-ray transition sequences
            for lvl in self.levels:              # Loop over levels/nodes in the level scheme space
                if TS.g.nodes[path[0]]['Gamma'] in self.levels[lvl].out_gammas: # If first transition in path is an outgoing gamma of Level 
                    start_lvl=lvl             # This is the starting level
                    #print('Start level: ',start_lvl)
                    break
            for lvl in self.levels:
                if TS.g.nodes[path[len(path)-1]]['Gamma'] in self.levels[lvl].in_gammas: # If last transition in path is incoming gamma...
                    stop_lvl=lvl                       # This is the stopping level
                    #print('Stop level: ',stop_lvl)
                    break
        
            if start_lvl not in self.src_to_dst:            # If the source level hasn't been added yet
                self.src_to_dst.update({start_lvl:[]})      # Update the dictionary
                self.src_to_dst[start_lvl].append(stop_lvl) # Add final level to list of possible destination levels
            else:                                    
                self.src_to_dst[start_lvl].append(stop_lvl)
        
            start_stop=(start_lvl,stop_lvl) # Declare tuple that specifies endpoints of transition pathway
            if start_stop not in self.cascade:   # If these end points have not been added to the cascade yet
                self.cascade.update({start_stop:[]})
                self.cascade[start_stop].append(path)
            else: # This tuple already exists in dictionary
                self.cascade[start_stop].append(path) # New possible pathway between source and destination levels
            
    def compute_path_lengths(self,energy_threshold=1):
        
        for start_stop in self.cascade: # Loop over all pairs of starting/stopping levels in cascade dictionary
            self.path_lengths.update({start_stop:[]})
            for pathway in self.cascade[start_stop]: # Loop over all pathways for a given pair of endpoints
                sumEnergy=0
                for gammaEnergy in pathway: # Loop over all gamma-rays within a given pathway
                    sumEnergy += gammaEnergy # Sum up the total energy between the source and destination levels
                self.path_lengths[start_stop].append(sumEnergy) # Add sum of gamma energies to the distance between two Levels
                
        for start_stop in self.path_lengths:
            for i in range(len(self.path_lengths[start_stop])-1):
                for j in range(i+1,len(self.path_lengths[start_stop])):
                    if abs(self.path_lengths[start_stop][i]-self.path_lengths[start_stop][j])>energy_threshold:
                        print('ERROR: path lenghts differ by more than energy threshold')
            sumEnergy=0
            for length in self.path_lengths[start_stop]:
                sumEnergy += length
            meanEnergy=sumEnergy/len(self.path_lengths[start_stop])
            self.path_lengths[start_stop].clear()
            self.path_lengths.update({start_stop:meanEnergy})        

    def merge_redundant_leaves(self,energy_threshold=1):
        
        updatedEdges = [] # List containing updated edges after nodes have been labeled as redundant
        gEnergies = []    # List of gamma energies that should be assigned to the new edges
        gWeights = [] # List of gamma intensities that should be assigned to the new edges
        
        for src in self.src_to_dst: # Loop over all starting Levels
        
            if len(self.src_to_dst[src])>1: # If the number of possible destination Levels is greater than one...
                for i in range(len(self.src_to_dst[src])-1): 
                    #print('Node i: ',src_to_dst[src][i])
                    if self.src_to_dst[src][i] in self.dummy_nodes: # If ith node has been labeled redundant, skip it
                        continue
                    for j in range(i+1,len(self.src_to_dst[src])):
                        #print('Node j: ',src_to_dst[src][j])
                        if self.src_to_dst[src][i] in self.dummy_nodes: # If jth node has been labeled redundant, skip it
                            continue
                        
            #if len(src_to_dst[src])==2:
                        dsti=self.src_to_dst[src][i] # ith leaf node
                        dstj=self.src_to_dst[src][j] # jth leaf node
                        if dsti==dstj: # Both paths lead to the same Level
                            #if lvlEnergies[(src,dsti)]!=lvlEnergies[(src,dstj)]: # If sum of gamma energies differ
                            if abs(self.path_lengths[(src,dsti)]-self.path_lengths[(src,dstj)])>energy_threshold:
                                print('ERROR: Gamma energies do not sum to same level energy!')
                            else: # If they're the same, nothing to see here. What you'd expect
                                continue
                        else: # Both paths lead to different nodes
                            #if lvlEnergies[(src,dsti)]!=lvlEnergies[(src,dstj)]: # If sum of gamma energies differ
                            #if abs(lvlEnergies[(src,dsti)]-lvlEnergies[(src,dstj)])>energy_threshold:
                            if abs(self.path_lengths[(src,dsti)]-self.path_lengths[(src,dstj)])>energy_threshold:
                                continue # This is what you'd expect. One of these levels isn't the ground state
                            else: # Nodes have the same sum of gamma energies
                                self.savedNodes.append(dsti) # List of nodes that duplicates but should be preserved
                                #if dstj not in dummy_nodes and dstj not in savedNodes:
                                if dstj not in self.dummy_nodes:
                                    self.dummy_nodes.append(dstj) # List of redundant nodes to be deleted
                                #for gamma in in_gammas[dstj]:
                                for gamma in self.levels[dstj].in_gammas:
                                    #if gamma not in in_gammas[dsti]:
                                    if gamma not in self.levels[dsti].in_gammas:
                                        #in_gammas[dsti].append(gamma)
                                        self.levels[dsti].in_gammas.append(gamma)
                                #else: # Already marked as a redundant node
                                #    continue
                                # Should be no outgoing gammas  in this leaf node
                                #for gamma in out_gammas[e1]:
                                #    if gamma not in out_gammas[e0]:
                                #        out_gammas[e0].append(gamma)
                                #for edge in ls.edges:
                                for edge in self.g.edges:
                                    #if edge[1]==dstj: # and dstj not in savedNodes: # If edge contains redundant leaf node...
                                    if edge[1]==dstj:
                                        if edge not in self.dummy_edges:
                                            self.dummy_edges.append(edge)
                                            #gamEn=ls[edge[0]][dstj]
                                            gamEn=self.g.edges[edge]['energy']
                                            #gamEn=gamEn['energy']
                                            gEnergies.append(gamEn)
                                            gamBR=self.g.edges[edge]['weight']
                                            gWeights.append(gamBR)
                                            newEdge=(edge[0],dsti)
                                            updatedEdges.append(newEdge)
                                #ls.add_edge(edge[0],dst0) # Make new edge connecting existing leaf node
                                            print('New edge: ',newEdge)
                                        else: # Already marked as redundant edge
                                            continue
                        #ls.remove_node(dst1) # Delete redundant node
                        #print('Redundant node removed: ',dst1) 
        print('Updated edges: ',updatedEdges)
        g=0 # Index counter for gamma-ray energies
        for e in updatedEdges:
            self.g.add_edge(e[0],e[1],energy=gEnergies[g],weight=gWeights[g])
            g+=1

                        
    def find_leaf_nodes(self):
        
        self.leaf_nodes.clear()
        for lvl in self.levels: # Loop over all Levels in the LevelScheme
            if len(self.levels[lvl].out_gammas)<1: # If the number of outgoing gammas from a level is zero...
                self.leaf_nodes.append(lvl) # Find its node number and add to the list of leaf nodes
                
    
    def leaf_node_deexcitation_energies(self):
        
        deexcitationEnergies = {}
        for node in self.leaf_nodes: # Loop over all potential ground states (i.e. leaf nodes)
            maxEnergy=0
            for start_stop in self.path_lengths: 
                if node==start_stop[1]: # If the stop node is a leaf node...
                    if self.path_lengths[start_stop]>maxEnergy: # Check if this is the largest deexcitation energy
                        maxEnergy=self.path_lengths[start_stop]
            deexcitationEnergies.update({start_stop:maxEnergy})
        print('Maximum energy lost when populating leaf nodes: ',deexcitationEnergies)
        
    def leaf_node_incoming_intensity(self,gammas,S):
        
        incomingIntensities = {}
        for node in self.leaf_nodes:
            gFlow=0
            for edge in self.g.edges:
                if edge[1]==node:
                    #gFlow+=self.g.edges[edge]['intensity']
                    gammaEnergy=self.g.edges[edge]['energy']
                    for gam in gammas: # Loop over all gammas
                        if gam.gE==gammaEnergy: # If the gamma energy corresponds to that of the edge connecting the leaf node
                            index=gammas.index(gam) # Get the index of this gamma-ray in the list
                            break
                    intensity=S[index] # Use the index of the gamma-ray of interest to get the intensity from Singles matrix
                    gFlow+=intensity # Add to the total gamma-ray flow into this level
            incomingIntensities.update({node:gFlow})
        print('Total number of gammas going into each leaf node: ',incomingIntensities)
    
    
    def find_ground_state(self,gammas,S):
        
        gamFlows={}
        for node in self.leaf_nodes:
            gFlow=0
            for edge in self.g.edges:
                if edge[1]==node:
                    #gFlow+=self.g.edges[edge]['intensity']
                    gammaEnergy=self.g.edges[edge]['energy']
                    for gam in gammas: # Loop over all gammas
                        if gam.gE==gammaEnergy: # If the gamma energy corresponds to that of the edge connecting the leaf node
                            index=gammas.index(gam) # Get the index of this gamma-ray in the list
                            break
                    intensity=S[index] # Use the index of the gamma-ray of interest to get the intensity from Singles matrix
                    gFlow+=intensity # Add to the total gamma-ray flow into this level
            gamFlows.update({node:gFlow})
        print(gamFlows)
        gs=max(gamFlows,key=gamFlows.get)
        print(gs)
        
    def find_root_nodes(self):
        
        self.root_nodes.clear()
        for lvl in self.levels: # Loop over all Levels in the LevelScheme
            if len(self.levels[lvl].in_gammas)<1: # If the number of incoming gammas to this Level is zero...
                self.root_nodes.append(lvl) # Find its node number and add to the list of root nodes
                
    def compute_level_energies(self,gs_node):
        
        self.g.nodes[gs_node]['energy']=0.0 # Set energy of the ground state node to 0.0 keV 
        self.levels[gs_node].update_excitation_energy(0.0) # Update ground-state Level's excitation energy to 0.0 keV
              
        eAssigned = [] # List of nodes that have been assigned energies
        eAssigned.append(gs_node) # Add ground-state node to the list
        iteration=0 # Counts the number of iterations in while loop
        maxIter=10  # Used as break condition to avoid infinite loop
        
        while len(eAssigned)<self.g.number_of_nodes():
            for edge in self.g.edges:
                if edge[1] in eAssigned and edge[0] not in eAssigned: # If lower level has assigned energy...
                    gamEn=self.g[edge[0]][edge[1]]['energy']    # Get gamma energy of associated edge
                    self.g.nodes[edge[0]]['energy']=self.g.nodes[edge[1]]['energy']+gamEn # Add gamma energy to lower level
                    self.levels[edge[0]].update_excitation_energy(self.g.nodes[edge[0]]['energy']) # Update Level energy
                    eAssigned.append(edge[0]) # Add to list of assigned Level energies
                elif edge[0] in eAssigned and edge[1] not in eAssigned: # If lower level has assigned energy...
                    gamEn=self.g[edge[0]][edge[1]]['energy']
                    self.g.nodes[edge[1]]['energy']=self.g.nodes[edge[0]]['energy']-gamEn # Subtract gamma energy from upper
                    self.levels[edge[1]].update_excitation_energy(self.g.nodes[edge[1]]['energy'])
                    eAssigned.append(edge[1]) # Add to list of assigned Level energies
                else: # Either both level energies have been assigned already or neither have
                    continue
            iteration+=1
            if iteration>maxIter:
                print('ERROR: Total number of iterations over graph has exceeded max allowed.')
                print('Total number of nodes: ',self.g.number_of_nodes())
                print('Number of assigned level energies: ',len(eAssigned))
                break
                
    def merge_redundant_roots(self,energy_threshold):
        
        updatedEdges = [] # List containing updated edges after nodes have been labeled as redundant
        gEnergies = []    # List of gamma energies that should be assigned to the new edges
        gWeights = [] # List of gamma intensities that should be assigned to the new edges
        
        for i in range(len(self.root_nodes)):         
    
            if self.root_nodes[i] in self.dummy_nodes:
                continue
            else:
                for j in range(i+1,len(self.root_nodes)):
                    if self.root_nodes[j] in self.dummy_nodes:
                        continue
                    elif abs(self.levels[self.root_nodes[i]].ExE-self.levels[self.root_nodes[j]].ExE)<energy_threshold:
                    #elif abs(ls.nodes[root_nodes[i]]['energy']-ls.nodes[root_nodes[j]]['energy'])<energy_threshold:
                        #savedNodes.append(root_nodes[i])
                        self.dummy_nodes.append(self.root_nodes[j])
                        for gamma in self.levels[self.root_nodes[j]].out_gammas:
                            self.levels[self.root_nodes[i]].out_gammas.append(gamma)
                            #for gamma in out_gammas[root_nodes[j]]:
                            #out_gammas[root_nodes[i]].append(gamma)
                        for edge in self.g.edges:
                            if edge[0]==rootNotes[j]:
                                self.dummy_edges.append(edge)
                                gamEn=self.g.edges[edge]['energy']
                                gEnergies.append(gamEn)
                                gamBR=self.g.edges[edge]['weight']
                                gWeights.append(gamBR)
                                newEdge=(root_nodes[i],edge[1])
                                updatedEdges.append(newEdge)
        
        g=0 # Index counter for gamma-ray energies/branching ratios
        for e in updatedEdges:
            self.g.add_edge(e[0],e[1],energy=gEnergies[g],weight=gWeights[g])
            g+=1
    
    # TODO write a cached_property resetter to update this whenever g is modified
    @cached_property
    def adj(self):
        A = np.zeros((self.nl, self.nl))
        for n in self.g:
            for nbr, datadict in self.g.adj[n].items():
                A[n, nbr] = datadict['weight']

        return A    

    def _add_out_transitions(self):

        # Add one transition from every level (except ground state)
        for i in reversed(range(1, self.nl)):
            nd = rnd.randint(i)

            # If energy levels of multiple nodes are equal,
            # we don't want to add a transition. Instead, we add
            # a transition down to the next highest energy
            while self.g.nodes[nd]['energy'] == self.g.nodes[i]['energy']:
                nd -= 1

            # Add an edge (transition) from high energy to low.
            # If this edge already exists, does nothing.
            self.g.add_edge(i, nd)

    def _add_in_transitions(self):

        # For every energy level that doesn't have a transition
        # into it from a higher energy, add a transition (except
        # for highest energy)
        for i in range(self.nl-1):
            if self.g.in_degree(i) == 0:
                nd = rnd.randint(i, high=self.nl)

                # Increase edge source until the source energy 
                # is greater than the energy at node 'i'
                while self.g.nodes[nd]['energy'] == self.g.nodes[i]['energy']:
                    nd += 1

                # Add edge from nd to i. If this edge already exists, 
                # this does nothing.
                self.g.add_edge(nd, i)
            
    def _add_rand_transitions(self):

        while self.g.number_of_edges() < self.nt:
            w = rnd.uniform(low=0.1, high=0.9)
            n1 = rnd.randint(self.nl-1)
            n2 = rnd.randint(self.nl-1)
            while n1 == n2:
                n2 = rnd.randint(self.nl-1)

            # Use elif statement so that no edge is added 
            # for equal energies
            if (self.g.nodes[n1]['energy'] > self.g.nodes[n2]['energy']):
                self.g.add_edge(n1, n2, weight=w)
            elif (self.g.nodes[n1]['energy'] < self.g.nodes[n2]['energy']):
                self.g.add_edge(n2, n1, weight=w)

    def _add_branch_probs(self):

        # Assign branching probabilities
        for n in self.g.nodes:
            probs_unnormalized = rnd.uniform(low=0.1, high=0.9, size=self.g.out_degree(n))
            probs = probs_unnormalized / np.sum(probs_unnormalized)

            for e, w in zip(self.g.out_edges(n), probs):
                self.g[e[0]][e[1]]['weight'] = w

    def make_random(self, num_levels, num_trans, EMax):

        self.nl = num_levels
        self.nt = num_trans
        self.emax = EMax

        # Create Nl different energy levels, with one guaranteed
        # to be 0. Then sort in ascencing order.
        energies = rnd.randint(0.01*EMax, high=EMax, size=self.nl-1)
        energies = np.append(energies, 0)
        energies = np.sort(energies)

        # Force the highest energy level to be unique
        if energies[-1] == energies[-2]:
            energies[-1] *= 1.2

        # Add each energy level as a node to the level graph
        for i, en in zip(range(self.nl), energies):
            self.g.add_node(i, energy = en)
            
        self._add_in_transitions()
        self._add_out_transitions()
        self._add_rand_transitions()
        self._add_branch_probs()
        
        return self.g


    def add_transition(self, initial, final, br):

        #if (self.g.nodes[n1]['energy'] > self.g.nodes[n2]['energy']):
        self.g.add_edge(initial,final, weight=br)
        self.nt=self.g.number_of_edges()
        #print('Initial: ',initial,'; Final: ',final)
        
        #maxE = max(out_gammas)
        #self.g.add_node(self.nl,maxE)
        
# ###################################################### # 
class TransitionScheme:
    
    def __init__(self):

        self.nt = 0           # Total number of transitions in scheme
        self.g = nx.DiGraph() # Directed graph object used to represent TransitionScheme
        self.adjacency={}       # Dictionary of adjacent gamma transitions
        self.nodeDict={}      # Dictionary where key is Node # and definition is Gamma object 
        self.leaf_nodes=[]     # List of all leaf nodes in the TransitionScheme
        self.branch_nodes=[]   # List of all branch nodes in the TransitionScheme
        self.gamma_energies=[]   # List of all gamma-ray energies; might replace this with list of Gamma objects
        self.all_paths=[]      # List of all possible transition pathways between two nodes in gamma cascade

    def __str__(self):

        return dedent(f"""
            Transition Scheme with {self.nt} transitions.""")
    
    def build_from_adjacency_matrix(self,adjMatrix,intensity_threshold,gammas):
        
        self.gamma_energies=gammas
        for i in range(len(adjMatrix[0])): # Loop over rows in adjacency matrix
            leaf=True # Is this a leaf node?
            for j in range(len(adjMatrix[0])): # Loop over columns in adjacency matrix
                
                if adjMatrix[i][j]<=intensity_threshold: # Matrix element not above threshold
                    continue # Keep looping
                else: # Adjacent transitions found!
                    leaf=False
                    if gammas[i].gE not in self.adjacency: # Gamma not placed in transition scheme yet
                        self.branch_nodes.append(gammas[i].gE)  # Add to list of branch nodes
                        self.adjacency.update({gammas[i].gE:[]}) # Add gamma energy to dictionary of adjacent transitions
                    self.adjacency[gammas[i].gE].append(gammas[j].gE) # Add jth gamma to ith gamma's list of adjacent transitions
                    self.g.add_node(gammas[i].gE,Gamma=gammas[i]) # Add ith node to directed graph of TransitionScheme
                    self.g.add_node(gammas[j].gE,Gamma=gammas[j]) # Add jth node to directed graph of TransitionScheme
                    self.g.add_edge(gammas[i].gE,gammas[j].gE,weight=adjMatrix[i][j])
            if leaf==True: # If after looping over all columns leaf is still true...
                self.leaf_nodes.append(gammas[i].gE) # Add to list of leaf nodes
                self.adjacency.update({gammas[i].gE:[]}) # Add gamma to dictionary
                self.g.add_node(gammas[i].gE,Gamma=gammas[i]) # Add ith node to directed graph of TransitionScheme
        print(self.adjacency)
        
    def find_all_paths(self,source,destination):
        # Clear previously stored paths
        path = []
        path.append(source)
        #print("Source : " + str(src) + " Destination : " +  str(dst))

        # Use depth first search (with backtracking) to find all the paths in the graph
        self.depth_first_search(source,destination,path)

        # Print all paths
        #self.Print ()
        
    def print_paths(self):
        # print (self.all_paths)
        for path in self.all_paths:
            print("Path : " + str(path))
        #self.all_paths.clear()
        
    # This function uses DFS at its core to find all the paths in a graph
    #def DFS (self, adjlist : Dict[int, List[int]], src : int, dst : int, path : List[int]):
    def depth_first_search(self,source,destination,path):
        if source==destination:
            self.all_paths.append(copy.deepcopy(path))
        else:
            for adjNode in self.adjacency[source]:
                path.append(adjNode)
                self.depth_first_search(adjNode,destination,path)
                path.pop()  

    @cached_property
    def adj(self):
        A = np.zeros((self.nt, self.nt))
        for n in self.g:
            for nbr, datadict in self.g.adj[n].items():
                A[self.g.nodes[n]['id'], self.g.nodes[nbr]['id']] = datadict['weight']

        return A   


    @classmethod
    def from_level_scheme(cls, lsc):
        new_tsc = TransitionScheme()
        for n in lsc.g:
            for nbr, datadict in lsc.g.adj[n].items():
                new_tsc.g.add_node((n, nbr), id=new_tsc.nt)
                new_tsc.nt += 1

        for u,v in new_tsc.g:
            for src, dest in lsc.g.out_edges(v):
                w = lsc.g[src][dest]['weight']
                new_tsc.g.add_edge((u,v), (src, dest), weight=w)
        
        return new_tsc