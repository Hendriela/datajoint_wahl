#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 16/05/2022 17:09
@author: hheise

Schema for histology analysis
"""

import numpy as np
import pandas as pd

import datajoint as dj
import login

login.connect()

from schema import common_mice

schema = dj.schema('common_hist', locals(), create_tables=True)


@schema
class Ontology(dj.Manual):
    definition = """ # Ontology tree of the Common Coordinate Framework of the Allen Mouse Brain Reference Atlas.
    structure_id        : int           # Unique structure ID, arbitrary order
    ----
    acronym             : varchar(16)   # Abbreviated unique structure name
    full_name           : varchar(128)  # Full (unique) structure name
    parent_id           : int           # Structure ID of the parent node
    depth               : tinyint       # Depth of this structure in the tree
    id_path             : varchar(64)   # Complete ID path, parent layers separated by '/'
    volume=NULL         : float         # Average volume of this structure in mm3. Some fine structures do not have volume data.
    volume_sd=NULL      : float         # Standard deviation of the average volume in mm3
    major_division      : tinyint       # Bool flag if the region is a major division (Isocortex, Hippocampal formation, etc.)
    summary_struct      : tinyint       # Bool flag if the region is a 'summary structure', a level useful for analyses
    color_hex           : char(6)       # Color hex code of the color this structure has in the ABA
    """

    def import_tree(self, filepath_api: str, filepath_vol: str) -> None:
        """
        Function to import the ontology data.

        The detailed ontology tree can be downloaded from the Allen Brain Atlas API:
        http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies
        Volume info for most of the larger structures is in the supplementary data of this publication:
        https://doi.org/10.1016/j.cell.2020.04.007, Table S4

        Both trees have nearly identical structure names and IDs, making integration of both relatively easy.
        However, three things have to be adjusted manually after downloading the files and before import:
        1. The acronym of the structure "cranial nerves" is renamed to "cnm" from the original "cm". This solves
            a duplicate primary key with the Central Medial Thalamic Nucleus, which has the acronym "CM", but
            DataJoint tables are by default case insensitive. This has to be changed in both files.
        2. Similarly, in the fine ontology, two regions have the acronyms "ipf" and "IPF". Change "ipf" to "ipff".
        2. The subregions of the Medial Mammillary Nucleus have the abbreviation "MM_" in the fine ontology, but "Mm_"
            in the volume tree. To avoid confusion, the fine ontology is changed to adapt the volume tree spelling.
        2. Similarly, the topmost layer is called "root" in the fine ontology. Change it to "Whole Brain" with "WB" as
            acronym for increased clarity.

        Args:
            filepath_api: Absolute path of the CSV file with the fine ontology tree from the API
            filepath_vol: Absolute path of the CSV file with the volume data from the publication
        """

        # filepath_api = r'W:\Neurophysiology-Storage1\Wahl\Datajoint\ABA_P56_ontology.csv'
        # filepath_vol = r'W:\Neurophysiology-Storage1\Wahl\Datajoint\ABA_volumes.csv'

        # Load CSV files
        ont_tree = pd.read_csv(filepath_api, header=0)
        vol_tree = pd.read_csv(filepath_vol, header=1)

        # Only take relevant columns from both tables and pattern-match entries by structure ID
        ont_tree_filt = ont_tree[['id', 'name', 'acronym', 'depth', 'structure_id_path', 'parent_structure_id',
                                  'color_hex_triplet']]
        vol_tree_filt = vol_tree[['structure ID', 'abbreviation', 'Major Division', 'Summary Structure',
                                  'Mean Volume (m)', 'Standard Deviation (s)']]
        merged = ont_tree_filt.merge(vol_tree_filt, how='left', left_on='id', right_on='structure ID')

        # Sanity check: are acronym and abbreviation identical?
        if np.sum(merged['acronym'] == merged['abbreviation']) != len(vol_tree_filt):
            raise ValueError('Merging of ontology tree with volume dataset failed!')

        # If the check passes, we dont need the duplicate columns anymore
        merged.drop(columns=['structure ID', 'abbreviation'], inplace=True)

        # Clean up boolean columns (not the volume columns, they have to show missing data for small regions)
        merged['Major Division'].fillna(0, inplace=True)
        merged['Summary Structure'].fillna(0, inplace=True)
        merged['Major Division'].replace('Y', 1, inplace=True)
        merged['Summary Structure'].replace('Y', 1, inplace=True)

        # Rename columns to the database attribute names
        merged.rename(columns={'id': 'structure_id', 'name': 'full_name', 'structure_id_path': 'id_path',
                               'parent_structure_id': 'parent_id', 'Mean Volume (m)': 'volume',
                               'Standard Deviation (s)': 'volume_sd', 'Major Division': 'major_division',
                               'Summary Structure': 'summary_struct', 'color_hex_triplet': 'color_hex'}, inplace=True)

        # Insert data row-wise into database
        connection = self.connection
        with connection.transaction:
            [self.insert1(dict(row)) for idx, row in merged.iterrows()]

    def map_acronym_to_id(self, invert: bool = False) -> dict:
        """
        Function that creates a dict mapping each acronym to its structure ID for rapid translation, or vice versa.

        Args:
            invert: Bool flag, if True it returns the ID as keys and acronyms as values instead.

        Returns:
            Mapping dictionary with acronym as key and its structure ID as value.
        """

        data = self.fetch('structure_id', 'acronym', as_dict=True)
        if invert:
            return {entry['structure_id']: entry['acronym'] for entry in data}
        else:
            return {entry['acronym']: entry['structure_id'] for entry in data}


@schema
class PrimaryAntibody(dj.Lookup):
    definition = """    # Different primary antibodies used in histology analysis
    target          : varchar(64)       # Name of the target protein
    primary_host    : varchar(32)       # Host species of the primary antibody.
    ---
    dilution        : int               # Diluting factor that has been proven to work best (written as 1:X)
    producer        : varchar(64)       # Company that produced the primary antibody
    description     : varchar(256)      # Information about the AB target
    """
    # add current licence, retrieve the licence file from the server
    contents = [
        ['GFAP', 'guinea pig', 750, 'Synaptic Systems',
         'Astrocyte marker, high expression in HPC and TH, less in cortex. Can mark lesions, where intensity should be increased.'],
        ['MAP2', 'rabbit', 250, 'Sigma-Aldrich',
         'Neuron-specific microtubule marker. Can mark lesions, where intensity should be decreased. Colocalizes with transgenic GFP, so better used in wildtype mice.'],
        ['intrinsic', 'intrinsic', 0, 'n.a.',
         'Intrinsic expression of fluorophore, through transgenic strains or viral injection.']
    ]


@schema
class Histology(dj.Manual):
    definition = """ # Data about the histology experiment
    -> common_mice.Mouse
    histo_date      : date                                          # Imaging date
    ----
    thickness       : int                                           # Slice thickness in um
    cutting_device  : enum('vibratome', 'cryostat')                 # Cutting device used
    direction       : enum('coronal', 'sagittal', 'horizontal')     # Cutting direction
    microscope      : varchar(256)                                  # Name of the microscope used for imaging
    """


@schema
class Staining(dj.Manual):
    definition = """    # Different primary fluorophores used during staining (or intrinsically expressed)
    -> Histology
    fluoro_num   : tinyint  # Number of the fluorophore in the histology experiment
    ---
    -> PrimaryAntibody      # Primary antibody used (NULL if fluorophore is expressed intrinsically, through strain or viral injection)
    fluorophore     : enum('GFP', 'Alexa488', 'Cy3', 'Alexa647', 'tdTomato')   # Fluorophore used to tag the primary antibody. All secondary ABs are from Jackson ImmunoResearch and used at 1:250 dilution.
    """
