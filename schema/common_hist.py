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
class ReferenceAtlas(dj.Manual):
    definition = """ # Annotated coronal reference atlas of the Common Coordinate Framework v3, used by the Allen Brain Atlas.
    image_id            : int           # ID of the image/atlas page, same as in the online interactive atlas
    ----
    bregma              : float         # distance from bregma of the slice in [mm], approximated from http://mouse.brain-map.org/experiment/siv/?imageId=102162070.
    atlas               : longblob      # 2D numpy array of the slice, each pixel containing the structure_id of the associated brain structure.
    resolution          : int           # Resolution of the atlas in [um/voxel]
    """

    def import_data(self, filepath: str, resolution: int) -> None:
        """
        Import a previously downloaded annotation atlas into the database.

        Args:
            filepath    : Absolute path to the atlas .npy file
            resolution  : Resolution of the downloaded atlas version
        """

        # Bregma values are copied from an old atlas because the current ABA does not use bregma as reference anymore
        bregma = [5.345, 5.245, 5.145, 5.045, 4.945, 4.845, 4.745, 4.645, 4.545, 4.445, 4.345, 4.245, 4.145, 4.045,
                  3.945, 3.845, 3.745, 3.645, 3.545, 3.445, 3.345, 3.245, 3.145, 3.045, 2.945, 2.845, 2.745, 2.645,
                  2.545, 2.445, 2.345, 2.245, 2.145, 2.045, 1.945, 1.845, 1.745, 1.645, 1.545, 1.445, 1.345, 1.245,
                  1.145, 1.045, 0.945, 0.845, 0.745, 0.645, 0.545, 0.445, 0.345, 0.245, 0.145, 0.02, -0.08, -0.18,
                  -0.28, -0.38, -0.48, -0.555, -0.655, -0.755, -0.855, -0.955, -1.055, -1.155, -1.255, -1.355, -1.455,
                  -1.555, -1.655, -1.755, -1.855, -1.955, -2.055, -2.155, -2.255, -2.355, -2.48, -2.555, -2.78, -2.88,
                  -2.98, -3.08, -3.18, -3.28, -3.38, -3.455, -3.58, -3.68, -3.78, -3.88, -3.98, -4.08, -4.18, -4.28,
                  -4.38, -4.455, -4.555, -4.655, -4.755, -4.855, -4.955, -5.055, -5.155, -5.255, -5.355, -5.455, -5.555,
                  -5.655, -5.755, -5.855, -5.955, -6.055, -6.18, -6.255, -6.355, -6.455, -6.555, -6.655, -6.755, -6.855,
                  -6.955, -7.055, -7.155, -7.255, -7.355, -7.455, -7.555, -7.655, -7.755, -7.905]

        # Annotated slices can be downloaded and saved with allen_api.py
        # filepath = r'W:\Neurophysiology-Storage1\Wahl\Datajoint\AllenAtlas\annotation_slices.npy'
        atlas = np.load(filepath)

        if len(bregma) != len(atlas):
            raise IndexError('The loaded atlas has a different number of slices than the Bregma values on record.')

        # Construct entries
        entries = [{'image_id': i, 'bregma': bregma[i], 'atlas': atlas[i],
                    'resolution': resolution} for i in range(len(bregma))]

        # Insert them into the database
        self.insert(entries)


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
