#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 16/05/2022 17:09
@author: hheise

Schema for histology analysis
"""
import datetime
from typing import Union, List

import numpy as np
import pandas as pd

import datajoint as dj
import login

login.connect()

from schema import common_mice
from util import helper

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

    def validate_grouping(self, group: List[list]) -> List[str]:
        """
        Validate a group of structures to find duplicates (every structure should be associated with a single grouping).
        Prints out duplicates if a structure is a child of multiple provided grouping structures.

        Args:
            group: List of lists of structure IDs or acronyms.

        Returns:
            List of top-most structure acronyms that are not included in any of the provided groups.
        """
        # If the grouping was given with acronyms, convert them to IDs
        if type(group[0][0]) != int:
            group_id = [[(self & f'acronym="{el}"').fetch1('structure_id') for el in sublist] for sublist in group]
        else:
            group_id = group

        # Flatten grouping
        group_flat = [element for sublist in group_id for element in sublist]
        group_flat_acr = [element for sublist in group for element in sublist]

        # Get data from database
        data = pd.DataFrame(self.fetch('acronym', 'structure_id', 'id_path', as_dict=True))
        group_flat_path = list((self & f'acronym in {helper.in_query(group_flat_acr)}').fetch('id_path'))

        # Order structures by increasing path length (to start of the topmost region)
        data_ordered = data.sort_values(by="id_path", key=lambda x: x.str.len())

        others = []
        others_path = []

        for path in data_ordered['id_path']:
            # Get parent structures of the current path
            parents = path.split('/')[1:-1]
            # Check if any grouped structure is a parent
            parent_in_group = [True if str(single_id) in parents else False for single_id in group_flat]

            # If there are more than one grouped structures which are parents, print them out
            if sum(parent_in_group) > 1:
                curr_struct = data[data["structure_id"] == int(parents[-1])]['acronym'].values[0]
                parent_structs = [group_flat_acr[i] for i in range(len(parent_in_group)) if parent_in_group[i]]
                print(f'\nStructure {curr_struct} is a child of multiple provided superstructures:\n{parent_structs}')

            # If the current structure is not included in any group...
            elif sum(parent_in_group) == 0:
                # First check if it is a parent structure of a group
                if not any(path in group_path for group_path in group_flat_path):
                    # If not, check if a parent of the current structure is already in the list (to only keep the topmost structure)
                    if not any(others_p in path for others_p in others_path):
                        curr_acr = data[data["structure_id"] == int(parents[-1])]['acronym'].values[0]
                        # Exclude unused structures 73 (ventricular system), 1024 (grooves),
                        # 549009199 (lateral strip of striatum) and 304325711 (retina)
                        if all(banned_id not in path for banned_id in ['73', '1024', '304325711', '549009199']):
                            # If not, we can append this region as a top-level "others" structure
                            others.append(curr_acr)
                            others_path.append(path)

        return others

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
    n_slices        : int                                           # Number of slices imaged, for whole-brain extrapolation
    rel_imaged_vol  : float                                         # Relative imaged volume of the whole brain
    cutting_device  : enum('vibratome', 'cryostat')                 # Cutting device used
    direction       : enum('coronal', 'sagittal', 'horizontal')     # Anatomical Cutting direction
    microscope      : varchar(256)                                  # Name of the microscope used for imaging
    """

    class HistoSlice(dj.Part):
        definition = """    # Different primary fluorophores used during staining (or intrinsically expressed)
        -> Histology
        glass_num       : int   # Number of the glass slide on which the slice is mounted (base 1)
        slice_num       : int   # Number of the slice on the slide (usually numbered from top left, base 0)
        ---
        slice_area      : float   # Area of the entire slice in [mm2]
        -> [nullable] ReferenceAtlas  # ID of the closest corresponding image in the annotation atlas (empty if not recorded)
        """

    def import_data(self, username: str, filepath: str, day: Union[str, datetime.date], thickness: int, cutting: str,
                    direction: str, microscope: str) -> None:
        """
        Import general information about a histology experiment, manually and from a CSV file.

        Args:
            username    : Name of the investigator to which the mouse/mice of the experiment belong.
            filepath    : Absolute file path to the CSV file holing slice surface areas of all imaged slices.
            day         : Day if the histology experiment, as a datetime.date object or string with format 'YYYY-MM-DD'.
            thickness   : Thickness of the slices, in um.
            cutting     : Cutting device/method used. Has to be either 'vibratome' or 'cryostat'.
            direction   : Anatomical cutting direction of the slices. Has to be 'coronal', 'sagittal' or 'horizontal'
            microscope  : Name of the microscope used for imaging.
        """

        # Load data from slice_size CSV file
        #
        slice_sizes = pd.read_csv(filepath)
        if len(slice_sizes.columns) == 4:
            # If the Image ID has not been tracked, the file should have 4 columns, and create an empty 5th
            slice_sizes.columns = ['mouse_id', 'glass_num', 'slice_num', 'slice_area']
            slice_sizes['image_id'] = np.nan
        elif len(slice_sizes.columns) == 5:
            slice_sizes.columns = ['mouse_id', 'glass_num', 'slice_num', 'slice_area', 'image_id']
        else:
            raise IndexError(f'Invalid number of columns: {len(slice_sizes.columns)}. File should have 4 or 5 columns.')

        # Safety check that all mice listed in the data exist in the database
        mouse_ids = slice_sizes['mouse_id'].unique()
        if not all([common_mice.Mouse & f'username="{username}"' & f'mouse_id={mouse_id}' for mouse_id in mouse_ids]):
            raise ImportError(f'Not all mice listed in {filepath} exist for user {username}.')

        # Convert area from um2 (QuPath) to mm2
        slice_sizes['slice_area'] = slice_sizes['slice_area']/1000000

        # Construct entries
        entries = []
        part_entries = []

        # Total brain volume is volume of "whole brain" minus volume of ventricular system (data from Allen Brain Atlas)
        brain_vol = (Ontology & 'parent_id=-1').fetch1('volume') - (Ontology & 'acronym="VS"').fetch1('volume')

        for mouse_id in mouse_ids:
            # Calculate total imaged volume and normalize it by the total brain volume
            imaged_vol = np.sum(thickness / 1000 * slice_sizes[slice_sizes['mouse_id'] == mouse_id]['slice_area'].sum())

            entries.append({'username': username, 'mouse_id': mouse_id, 'histo_date': day, 'thickness': thickness,
                            'n_slices': len(slice_sizes[slice_sizes['mouse_id'] == mouse_id]),
                            'rel_imaged_vol': imaged_vol/brain_vol, 'cutting_device': cutting, 'direction': direction,
                            'microscope': microscope})
            for idx, row in slice_sizes[slice_sizes['mouse_id'] == mouse_id].iterrows():
                part_entries.append({'username': username, 'histo_date': day, **row})

        # Insert entries into database
        connection = self.connection
        with connection.transaction:
            self.insert(entries)
            self.HistoSlice().insert(part_entries)


@schema
class Staining(dj.Manual):
    definition = """    # Different primary fluorophores used during staining (or intrinsically expressed)
    -> Histology
    fluoro_num   : tinyint  # Number of the fluorophore in the histology experiment
    ---
    -> PrimaryAntibody      # Primary antibody used (NULL if fluorophore is expressed intrinsically, through strain or viral injection)
    fluorophore     : enum('GFP', 'Alexa488', 'Cy3', 'Alexa647', 'tdTomato')   # Fluorophore used to tag the primary antibody. All secondary ABs are from Jackson ImmunoResearch and used at 1:250 dilution.
    """
