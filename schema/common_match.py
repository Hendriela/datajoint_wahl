#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13/01/2022 18:03
@author: hheise

Schema for tracking and matching single neurons across sessions
"""
import numpy as np
from skimage.draw import polygon
from skimage import measure
from scipy import spatial
from typing import Iterable, Tuple, List
import math
import bisect

import datajoint as dj

from schema import common_img

schema = dj.schema('common_match', locals(), create_tables=True)


@schema
class CellMatchingParameter(dj.Manual):
    definition = """ # Parameters used in the cross-session cell matching pipeline. Defaults are params of Annas model.
    matching_param_id           : int       # Index of parameter set
    ------
    contour_thresh = 0.05       : float     # Intensity threshold of ROI contour finding
    true_binary = 0             : tinyint   # Bool flag whether the footprint should truly be binarized or areas outside 
                                            # the contour but with weight > 0 should retain their non-zero weights.
    neighbourhood_radius = 80   : int       # Radius in um of the area around the ROI where matches are considered. 
                                            # Default of 80 translates to 50 pixels at 1x zoom
    nearby_neuron_cap = 15      : int       # Maximum nearest neurons found by KDTree nearest-neighbor-analysis
    """


@schema
class CellMatchingClassifier(dj.Manual):
    definition = """ # Table to store decision tree classifier models for cell matching GUI
    classifier_id   : int           # Index of classifier version
    ------
    description                     : varchar(256)  # Description of the model
    model_path                      : varchar(256)  # Path to classifier model, relative to Wahl folder server 
    train_time = CURRENT_TIMESTAMP  : timestamp     # Timestamp of the training of the model
    """


@schema
class MatchingFeatures(dj.Computed):
    definition = """ # Features of ROIs that are used as input for the cell matching classifier
    -> common_img.Segmentation
    -> CellMatchingParameter
    ------
    match_time = CURRENT_TIMESTAMP  : timestamp
    """

    class ROI(dj.Part):
        definition = """ # Features of neurons that are used as input for the cell matching classifier
        -> master
        mask_id                     : int           # ID of the ROI (same as common_img.Segmentation)
        ------
        contour                     : longblob      # (Semi)-binarized spatial contour of the ROI
        neighbourhood               : longblob      # Local area crop of the local correlation template around the neuron
        rois_nearby                 : int           # Number of neurons in the neighbourhood. Capped by parameter.
        closest_roi                 : int           # Index of the nearest ROI (smallest physical distance)
        closest_roi_angle           : float         # Radial angular distance of the closest ROI
        neighbours_quadrant         : longblob      # Number of nearby ROIs split into quadrants (clockwise, from top-left)
        """

    def make(self, key: dict) -> None:
        """
        Compute ROI features that are used as criteria for matching cells across sessions (through Anna's GUI).

        Args:
            key: Primary keys of the current MatchingFeatures() entry.
        """

        # Fetch relevant data
        footprints = (common_img.Segmentation.ROI & key).get_rois()
        coms = np.vstack((common_img.Segmentation.ROI & key).fetch('com'))
        template = (common_img.QualityControl & key).fetch1("cor_image")
        params = (CellMatchingParameter & key).fetch1()

        # Convert neighbourhood radius (in microns) to zoom-dependent radius in pixels
        mean_res = np.mean((common_img.CaimanParameter & key).get_parameter_obj(key)['dxy'])
        margin_px = int(np.round(params['neighbourhood_radius']/mean_res))

        coms_list = [list(com) for com in coms]     # KDTree expects a list of lists as input
        neighbor_tree = spatial.KDTree(coms_list)   # Build kd-tree to query nearest neighbours of any ROI

        new_entries = []
        for roi_idx in range(coms.shape[0]):
            com = coms[roi_idx]

            # Crop and binarize current footprint
            footprint = self.binarize_footprint(footprints[roi_idx], params['contour_thresh'],
                                                true_binary=params['true_binary'])

            # Crop the template around the current ROI
            area_image = self.crop_template(template, com, margin_px)[0]

            # Use KDTree's nearest-neighbour analysis to get the k closest ROIs
            distance, index = neighbor_tree.query(coms_list[roi_idx], k=params['nearby_neuron_cap'])

            # Get the index and radial angular distance of the nearest ROI
            closest_idx = index[1]
            closest_roi_angle = math.atan2(com[1] - coms[closest_idx][1], com[0] - coms[closest_idx][0])

            # get the number of nearest neighbours in the neighbourhood and their indices
            num_neurons_in_radius = bisect.bisect(distance, 50) - 1          # -1 to not count the ROI itself
            index_in_radius = index[1: max(0, num_neurons_in_radius) + 1]    # start at 1 to not count the ROI itself

            # Get the number of neighbours in each neighbourhood quadrant (top left to bottom right)
            neighbours_quadrants = self.neighbours_in_quadrants(coms, roi_idx, index_in_radius)

            # Create part-entry
            neuron_features = {'contour': footprint, 'neighbourhood': area_image,
                               'closest_roi': closest_idx, 'closest_roi_angle': closest_roi_angle,
                               'rois_nearby': num_neurons_in_radius,
                               'neighbours_quadrant': neighbours_quadrants}
            new_entries.append(dict(**key, **neuron_features, mask_id=roi_idx))

        # After all ROIs have been processed, insert master and part entries
        self.insert1(key)
        self.ROI().insert(new_entries)

    @staticmethod
    def binarize_footprint(footprint: np.ndarray, contour_thresh: float, true_binary: bool) -> np.ndarray:
        """
        Crops a spatial footpring to its minimal rectangle and binarizes the footprint with a given threshold.
        As cropping happens before thresholding, the final output array may be larger than the thresholded footprint.

        Args:
            footprint: 2D array with shape (x_fov, y_fov) of the FOV with the spatial footprint weights
            contour_thresh: Threshold of the contour finding algorithm which is applied to the spatial weights
            true_binary: Bool flag whether the image should truly be binarized or areas that are outside the contour
                            but still have a weight > 0 should retain their non-zero weights.

        Returns:
            2D array with shape (x_crop, y_crop) with the cropped and binarized footprint.
        """
        # crop FOV to the minimal rectangle of the footprint
        coords_non0 = np.argwhere(footprint)
        x_min, y_min = coords_non0.min(axis=0)
        x_max, y_max = coords_non0.max(axis=0)
        cropped_footprint = footprint[x_min:x_max + 1, y_min:y_max + 1]

        # measure contour area
        contours = measure.find_contours(cropped_footprint, contour_thresh, fully_connected='high')

        # Todo: validate that the actual binary image works as well as the semi-binarized version
        if true_binary:
            new_footprint = np.zeros(cropped_footprint.shape, dtype=np.int8)
        else:
            new_footprint = np.copy(cropped_footprint)

        if contours:
            # Compute polygon area of the contour and fill it with 1
            rr, cc = polygon(contours[0][:, 0], contours[0][:, 1], cropped_footprint.shape)
            if true_binary:
                new_footprint[rr, cc] = 1
            else:
                new_footprint[rr, cc] = 255

        else:
            # if no contour could be found, binarize the entire footprint manually
            print("no contour")
            new_footprint = np.where(cropped_footprint > contour_thresh, 1, 0)

        return new_footprint

    @staticmethod
    def crop_template(template: np.ndarray, roi_com: Iterable, margin_px: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crops the template FOV around the center of mass of an ROI.

        Args:
            template    : 2D array, FOV in which the ROI is located
            roi_com     : X and Y coordinates of the center of mass of the ROI
            margin_px   : Radius of the cropped image, in pixel

        Returns:
            Cropped FOV, and new CoM coordinates relative to the cropped FOV
        """
        # Translate the margin in um into pixels through the mean resolution (x and y resolution differ slightly)

        ymin = max(0, int(roi_com[1] - margin_px))
        ymax = min(template.shape[1], int(roi_com[1] + margin_px))
        xmin = max(0, int(roi_com[0] - margin_px))
        xmax = min(template.shape[0], int(roi_com[0] + margin_px))
        cropped = template[xmin:xmax, ymin:ymax]
        return cropped, (roi_com[1] - ymin, roi_com[0] - xmin)

    @staticmethod
    def neighbours_in_quadrants(coms: np.ndarray, neuron_ref: int, neuron_idxs: np.ndarray) -> List[int]:
        """ Calculates the number of ROIs that are in each quadrant of the reference ROI (excluding itself).

        Args:
            coms: 2D array of shape (n_neurons, 2) with center of mass of all ROIs in the FOV.
            neuron_ref: Index of the reference ROI.
            neuron_idxs: Indices of the ROI that should be sorted into quadrants (with respect to the ref neuron).

        Returns:
            List with four integers, the number of ROIs in each quadrant of the reference ROI, in order top-left, top-
                right, bottom-left, bottom-right.
        """
        quadrant_split = [0, 0, 0, 0]
        x_ref = coms[neuron_ref][0]
        y_ref = coms[neuron_ref][1]
        for idx in neuron_idxs:
            x = coms[idx][0]
            y = coms[idx][1]
            if x_ref <= x and y_ref <= y:
                quadrant_split[0] += 1      # top-left quadrant
            elif x_ref <= x and y_ref >= y:
                quadrant_split[1] += 1      # top-right quadrant
            elif x_ref >= x and y_ref >= y:
                quadrant_split[2] += 1      # bottom-left quadrant
            else:
                quadrant_split[3] += 1      # bottom-right quadrant
        return quadrant_split


@schema
class MatchedIndex(dj.Manual):
    definition = """ # Matched indices of ROIs in other sessions, created through Cell Matching GUI
    -> MatchingFeatures.ROI
    matched_session : varchar(64)   # Identifier for the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid
    ------
    matched_id      : int           # Mask ID of the same neuron in the matched session
    """
