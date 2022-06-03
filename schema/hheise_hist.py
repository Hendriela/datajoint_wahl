#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 17/05/2022 12:53
@author: hheise

Histology analysis for microsphere model.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

import datajoint as dj
import login

login.connect()

from schema import common_hist
from util import helper

schema = dj.schema('common_hist', locals(), create_tables=True)


@schema
class Microsphere(dj.Manual):
    definition = """ # Quantification of microsphere histology, each entry is data from one sphere/lesion. Data imported from CSV file.
    -> common_hist.Histology.HistoSlice
    hemisphere      : tinyint       # Bool flag if the sphere/lesion is in the left (=1, ipsilateral to window) or right (=0) hemisphere
    -> common_hist.Ontology         # Acronym of the structure/area of the sphere/lesion
    lesion          : tinyint       # Bool flag if the spheres are associated with visible damage or lesion
    ----
    spheres         : int           # Number of spheres in this area. Can be 0 if a clear lesion has no spheres around.
    map2=NULL       : float         # Area of damage [mm2] in the MAP2 staining. 0 if no visible damage, and NULL if MAP2 was not stained for.
    gfap=NULL       : float         # Area of damage [mm2] in the GFAP staining. 0 if no visible damage, and NULL if GFAP was not stained for.
    auto=NULL       : float         # Area of damage [mm2] in the autofluorescence. 0 if no visible damage, and NULL if autofluorescence was not analyzed.
    """

    def import_from_csv(self, username: str, filepath: str, hist_date: str) -> None:
        """
        Function to import sphere annotation data from a CSV file. The file has to have the following columns, in this
        order: mouse_id - glass - slice - hemisphere - area acronym - spheres - lesion - map2 - gfap - auto.

        Args:
            username: Shortname of the investigator
            filepath: Absolute path of the CSV file
            hist_date: Date of the imaging, in format 'YYYY-MM-DD'
        """

        # Load annotation data
        annot = pd.read_csv(filepath)

        # Rename columns to make consistent with database
        annot.columns = ['mouse_id', 'glass_num', 'slice_num', 'hemisphere', 'acronym', 'spheres', 'lesion', 'map2',
                         'gfap', 'auto']
        col = annot.columns

        # Check if any staining is missing completely (will be dropped from DataFrame and NULLed during insert)
        bad_cols = [col[i] for i in range(7, 10) if pd.isna(annot[col[i]]).all()]
        annot.drop(columns=bad_cols, inplace=True)
        col = annot.columns

        ### DATA INTEGRITY CHECKS ###
        # do all rows with "lesion=1" have an area datapoint, and vice versa?
        lesions = annot[annot[col[6]] == 1]
        bad_rows = np.where(lesions[col[7:]].isnull().apply(lambda x: all(x), axis=1))[0]
        if len(bad_rows):
            print('The following rows are marked as "damage", but have no damaged area on record:')
            for row in bad_rows:
                print(lesions.iloc[row], '\n')
            raise ValueError

        healthy = annot[annot[col[6]] == 0]
        no_lesion_rows = np.where(healthy[col[7:]].isnull().apply(lambda x: all(x), axis=1))[0]
        bad_rows = np.where(np.isin(np.arange(len(healthy)), no_lesion_rows, invert=True))[0]
        if len(bad_rows):
            print('The following rows are marked as "healthy", but have damaged areas on record:')
            for row in bad_rows:
                print(healthy.iloc[row], '\n')
            raise ValueError

        # Do all structure acronyms exist in the ontology tree in the database?
        bad_structs = [entry for idx, entry in annot.iterrows() if
                       len(common_hist.Ontology() & f'acronym="{entry["acronym"]}"') == 0]
        if len(bad_structs):
            print('The following rows are from invalid structures. Check spelling:')
            for row in bad_structs:
                print(row, '\n')
            raise ValueError
        #############################

        # Transform remaining NaNs to 0
        annot = annot.fillna(0)

        # Convert lesion size from um2 (used by QuPath) to mm2 (used during analysis)
        for data_col in col[7:]:
            annot[data_col] /= 1000000

        # Translate acronym to structure ID, which is used in the database
        mapping = common_hist.Ontology().map_acronym_to_id()

        # Transform data into single-entry dicts and insert into database
        with self.connection.transaction:
            for idx, row in annot.iterrows():
                entry = dict(username=username, histo_date=hist_date, **row)
                try:
                    entry['structure_id'] = mapping[row['acronym']]
                    del entry['acronym']
                except KeyError:
                    raise KeyError(f'Could not find acronym {row["acronym"]} of entry {entry}.')
                self.insert1(entry)








@schema
class MicrosphereSummary(dj.Computed):
    definition = """ # Some summary results about the whole brain (combined data from all imaged slices of one mouse).
    -> common_hist.Histology
    ----
    time_ana = CURRENT_TIMESTAMP    : timestamp     # automatic timestamp
    """

    class Metric(dj.Part):
        definition = """ # Summary results of an individual metric (e.g. spheres, autofluorescence damage, etc.).
        -> MicrosphereSummary
        metric_name     : varchar(32)   # Name of the metric (spheres, map2, gfap, auto)
        ----
        count           : float     # Number of spheres/volume of damage [mm3] in all slices
        count_extrap    : float     # Extrapolated number of spheres/volume based on percentage of imaged brain volume.
        num_slices      : int       # Number of slices in which the metric has data points.
        """

    def make(self, key: dict) -> None:

        # Fetch relevant data
        # for mouse in np.unique(common_hist.Histology().fetch('mouse_id')):
        #     key = dict(mouse_id=mouse)
        data = pd.DataFrame((Microsphere & key).fetch())

        # Calculate absolute imaged volume from slice thickness and surface
        thickness, total_slices = (common_hist.Histology & key).fetch1('thickness', 'n_slices')
        surface = (common_hist.Histology.HistoSlice & key).fetch('slice_area')
        imaged_vol = np.sum(thickness / 1000 * surface)
        # Total brain volume is volume of "whole brain" minus volume of ventricular system (data from Allen Brain Atlas)
        brain_vol = (common_hist.Ontology & 'parent_id=-1').fetch1('volume') - \
                    (common_hist.Ontology & 'acronym="VS"').fetch1('volume')
        # Divide imaged volume by total brain volume for relative imaged volume
        perc_vol = imaged_vol / brain_vol
        # print(mouse, '-', imaged_vol)

        # Compute data for spheres and lesions
        entries = []
        for metric in ['spheres', 'lesion_spheres', 'map2', 'gfap', 'auto']:
            if metric == 'lesion_spheres':
                entry = dict(**key, metric_name=metric)
                only_lesion = data[data['lesion'] > 0]
                entry['count'] = only_lesion['spheres'].sum()
                entry['count_extrap'] = entry['count'] * 1 / perc_vol
                entry['num_slices'] = len(only_lesion[only_lesion['spheres'] > 0].drop_duplicates(subset=['histo_date',
                                                                                                          'glass_num',
                                                                                                          'slice_num']))
                entries.append(entry)
            elif not all(data[metric].isna()):
                entry = dict(**key, metric_name=metric)
                entry['count'] = data[metric].sum()
                entry['count_extrap'] = entry['count'] * 1 / perc_vol
                entry['num_slices'] = len(data[data[metric] > 0].drop_duplicates(subset=['histo_date', 'glass_num',
                                                                                         'slice_num']))
                entries.append(entry)

        # Insert entries into database
        self.insert1(key)
        self.Metric().insert(entries)
