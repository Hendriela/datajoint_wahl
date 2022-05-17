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

schema = dj.schema('common_hist', locals(), create_tables=True)


@schema
class Ontology(dj.Manual):
    definition = """ # Ontology tree of the Common Coordinate Framework of the Allen Mouse Brain Reference Atlas.
    acronym             : varchar(16)   # Abbreviated unique structure name
    ----
    structure_id        : int           # Unique structure ID, arbitrary order
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


