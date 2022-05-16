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
    structure_id    : int           # Unique structure ID, arbitrary order
    ----
    order           : int           # Order given by Allen institute. More or less following AP axis.
    full_name       : varchar(128)  # Full (unique) structure name
    abbr            : varchar(16)   # Abbreviated (unique) structure name
    parent_id       : int           # Structure ID of the parent node
    depth_in_tree   : tinyint       # Depth of this region in the tree
    id_path         : varchar(64)   # Complete ID path, regions separated by '/'
    voxels          : int           # Total voxel count (10 um edge length) of the region and its subregions
    independent     : tinyint       # Bool flag if the region does not consist of subregions (independently delineated)
    major_division  : tinyint       # Bool flag if the region is a mayor division (Isocortex, Hippocampal formation, etc.)
    summary_struct  : tinyint       # Bool flag if the region is a 'summary structure', a level useful for analyses
    """

    def import_tree(self, filepath: str) -> None:
        """
        Function to import the tree file from https://doi.org/10.1016/j.cell.2020.04.007 into the database.

        Args:
            filepath: Absolute path of the CSV file
        """

        # Load CSV file
        tree = pd.read_csv(filepath, header=1)

        # Clean up data
        tree = tree.fillna(0)                       # Turn NaNs into 0
        tree = tree.astype({'parent_id': int})      # Fix datatype
        tree.replace('Y', 1, inplace=True)          # Make boolean columns binary
        tree.rename(columns={'structure ID': 'structure_id', 'full structure name': 'full_name',
                             'abbreviation': 'abbr', 'depth in tree': 'depth_in_tree',
                             'structure_id_path': 'id_path', 'total_voxel_counts (10 um)': 'voxels',
                             'Structure independently delineated (not merged to form parents)': 'independent',
                             'Major Division': 'major_division',
                             '"Summary Structure" Level for Analyses': 'summary_struct'}, inplace=True)

        # Insert data row-wise into database
        [self.insert1(dict(row)) for idx, row in tree.iterrows()]

