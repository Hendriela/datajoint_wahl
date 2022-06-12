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

        # Filter out annotated lesions in certain lateral thalamic nuclei which are not associated with spheres.
        # Damage in these areas is most likely caused by the window, not the spheres, as it can be seen also in mice without spheres.
        filter_structs = ['LP', 'LD', 'RT', 'LGd', 'LGv']
        only_structs = annot['acronym'].isin(filter_structs)
        only_no_spheres = annot['spheres'] == 0
        only_lesions = annot['lesion'] == 1
        combine_filter = only_structs & only_no_spheres & only_lesions
        annot = annot[~combine_filter]

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

    def get_structure(self, structure: Union[str, int], histo_key: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """
        Get processed histology data of all mice of a specific structure. The function queries data from
        Microspheres() and combines data of the provided structure and all of its subregions.

        Args:
            structure: Structure ID, Acronym or full name of the target structure.
            histo_key: Optional, primary keys of a specific histology experiment. If None, the whole table will be queried.

        Returns:
            DataFrame with one entry per mouse and the following columns:
                - spheres_total (total number of spheres in the structure)
                - spheres_lesion (number of spheres which are associated with a lesion)
                - lesion (Volume of damaged area in each of the three channels (auto, gfap, map2). If a channel was not
                    imaged, the columns do not exist.
                - lesion_spheres (Volume of damaged area that is associated with spheres)
                - xxx_rel (to all of these columns, this column normalizes the value to the entire brain. Eg. a auto_rel
                    value of 0.2 means that 20% of the total autofluorescence damage in this animal was found in the
                    current structure. A spheres_lesion_rel value of 0.2 means that 20% of spheres which are associated
                    with lesions are found in this structure.)
            If no mouse has data for the given structure, return None.
        """

        def summarize_single_metric(series: pd.Series, global_val: float, h: Optional[int]) -> Tuple[float, float]:
            """
            Calculate absolute and relative summary of a data series (a specific metric/channel).
            Args:
                series: Data of the metric, column of "curr_mouse" DataFrame (sphere count or lesion area in mm2).
                global_val: Value of the metric in the whole brain for normalization.
                h: Height/thickness of the sample, in um. Set to None for non-volume metrics (sphere counts).

            Returns:

            """
            if h is None:
                summed = series.sum()
            else:
                summed = series.sum() * (h / 1000)

            # If a metric is 0 in the entire imaged tissue, set rel_summed manually to 0 to avoid NaN
            if global_val > 0:
                rel_summed = summed / global_val
            else:
                rel_summed = 0.0
            return summed, rel_summed

        # If restriction is given, apply it to the queried histology experiment
        if histo_key:
            data = self & histo_key
        else:
            data = self

        # Get ID of the structure, if not given
        if type(structure) != int:
            try:
                id = (common_hist.Ontology & f'acronym="{structure}"').fetch1('structure_id')
            except dj.errors.DataJointError:
                try:
                    id = (common_hist.Ontology & f'full_name="{structure}"').fetch1('structure_id')
                except dj.errors.DataJointError as ex:
                    raise dj.errors.DataJointError(f'\nCould not interpret structure "{structure}". Use either the '
                                                   f'ID, acronym or full name of a structure.\nError:\n{ex}')
        else:
            id = structure

        # Raise warning if the selected structure does not have a volume on record
        if np.isnan((common_hist.Ontology & f'structure_id={id}').fetch1('volume')):
            raise UserWarning(f'Selected structure {structure} (ID {id}) has no volume on record. Relative results '
                              f'cannot be computed.')

        # Select only data from regions that have the structure ID in their ID path
        query = (data * common_hist.Ontology) & f'id_path like "%/{id}/%"'
        if len(query) == 0:
            # raise dj.errors.QueryError(f'Query for structure ID {id} returned no data.')
            return None

        # Process data to for individual mice
        results = {}
        for mouse in np.unique(query.fetch('mouse_id')):
            # Restrict for current mouse and fetch data
            curr_mouse = pd.DataFrame((query & f'mouse_id={mouse}').fetch())

            if len(curr_mouse['username'].unique()) > 1:
                raise NotImplementedError('More than 1 user queried, not implemented yet.')
            elif len(curr_mouse['histo_date'].unique()) > 1:
                raise NotImplementedError('More than 1 histology session queried, not implemented yet.')

            entry_key = dict(username=curr_mouse['username'].unique()[0], mouse_id=mouse,
                             histo_date=curr_mouse['histo_date'].unique()[0])

            curr_results = {}

            # # Get primary keys for unique slices (first 5 columns) to efficiently query slice areas
            # slice_keys = curr_mouse.drop_duplicates(subset=['histo_date', 'glass_num', 'slice_num']).iloc[:, :5]
            # areas = (common_hist.Histology.HistoSlice & slice_keys)

            # Get metrics of the whole dataset
            thickness = (common_hist.Histology & entry_key).fetch1('thickness')
            metrics = np.unique((MicrosphereSummary.Metric & entry_key).fetch('metric_name'))
            global_data = {}
            for metric in metrics:
                try:
                    global_data[metric] = (MicrosphereSummary.Metric & entry_key & f'metric_name="{metric}"').fetch1(
                        'count')
                except dj.errors.DataJointError:
                    pass

            ### Get aggregate results of sphere counts and lesion volume ###
            # Total number of spheres found in this structure
            curr_results['spheres_total'], curr_results['spheres_rel'] = \
                summarize_single_metric(curr_mouse['spheres'], global_data['spheres'], None)

            # Number of these spheres that were associated with a lesion
            curr_results['spheres_lesion'], curr_results['spheres_lesion_rel'] = \
                summarize_single_metric(curr_mouse[curr_mouse['lesion'] == 1]['spheres'],
                                        global_data['spheres_lesion'], None)

            # Volume of MAP2/GFAP/autofluorescence damage in mm3 in this structure
            if not all(curr_mouse['map2'].isna()):
                curr_results['map2'], curr_results['map2_rel'] = \
                    summarize_single_metric(curr_mouse['map2'], global_data['map2'], thickness)
                curr_results['map2_spheres'], curr_results['map2_spheres_rel'] = \
                    summarize_single_metric(curr_mouse[curr_mouse['spheres'] > 0]['map2'],
                                            global_data['map2_spheres'], thickness)

            if not all(curr_mouse['gfap'].isna()):
                curr_results['gfap'], curr_results['gfap_rel'] = \
                    summarize_single_metric(curr_mouse['gfap'], global_data['gfap'], thickness)
                curr_results['gfap_spheres'], curr_results['gfap_spheres_rel'] = \
                    summarize_single_metric(curr_mouse[curr_mouse['spheres'] > 0]['gfap'],
                                            global_data['gfap_spheres'], thickness)

            if not all(curr_mouse['auto'].isna()):
                curr_results['auto'], curr_results['auto_rel'] = \
                    summarize_single_metric(curr_mouse['auto'], global_data['auto'], thickness)
                curr_results['auto_spheres'], curr_results['auto_spheres_rel'] = \
                    summarize_single_metric(curr_mouse[curr_mouse['spheres'] > 0]['auto'],
                                            global_data['auto_spheres'], thickness)
            results[mouse] = curr_results

        return pd.DataFrame(results).T


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
        """
        Calculate overview stats for the microsphere annotation. Summarize recorded spheres and lesions across slices.

        Args:
            key: Primary keys for the current common_hist.Histology() entry.
        """
        # Fetch relevant data
        data = pd.DataFrame((Microsphere & key).fetch())
        perc_vol = (common_hist.Histology & key).fetch1('rel_imaged_vol')
        thickness = (common_hist.Histology & key).fetch1('thickness') / 1000    # slice thickness in mm

        # Compute data for spheres and lesions
        entries = []
        for metric in ['spheres', 'map2', 'gfap', 'auto']:
            if not all(data[metric].isna()):
                # Total count
                entry = dict(**key, metric_name=metric)
                entry['count'] = data[metric].sum()
                # For all metrics except the sphere count, we have to convert the imported areas in mm2 to volume in mm3
                if metric != 'spheres':
                    entry['count'] *= thickness
                entry['count_extrap'] = entry['count'] / perc_vol
                entry['num_slices'] = len(data[data[metric] > 0].drop_duplicates(subset=['histo_date', 'glass_num',
                                                                                         'slice_num']))
                entries.append(entry)

                # Spheres associated with lesions
                if metric == 'spheres':
                    entry = dict(**key, metric_name='spheres_lesion')
                    only_lesion = data[data['lesion'] > 0]
                    entry['count'] = only_lesion[metric].sum()
                    entry['count_extrap'] = entry['count'] * 1 / perc_vol
                    entry['num_slices'] = len(
                        only_lesion[only_lesion[metric] > 0].drop_duplicates(subset=['histo_date', 'glass_num',
                                                                                        'slice_num']))
                # Lesions associated with spheres
                else:
                    entry = dict(**key, metric_name=metric+'_spheres')
                    only_spheres = data[data['spheres'] > 0]
                    entry['count'] = only_spheres[metric].sum() * thickness     # again, convert area to volume
                    entry['count_extrap'] = entry['count'] * 1 / perc_vol
                    entry['num_slices'] = len(
                        only_spheres[only_spheres[metric] > 0].drop_duplicates(subset=['histo_date', 'glass_num',
                                                                                       'slice_num']))
                entries.append(entry)

        # Insert entries into database
        self.insert1(key)
        self.Metric().insert(entries)
