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
from typing import Iterable, Tuple, List, Optional
import math
import bisect
import ctypes

import datajoint as dj

from schema import common_img

schema = dj.schema('common_match', locals(), create_tables=True)


@schema
class CellMatchingParameter(dj.Manual):
    definition = """ # Parameters used in the cross-session cell matching pipeline. Defaults are params of Annas model.
    match_param_id              : int       # Index of parameter set
    ------
    contour_thresh = 0.05       : float     # Intensity threshold of ROI contour finding
    true_binary = 0             : tinyint   # Bool flag whether the footprint should truly be binarized or areas outside the contour but with weight > 0 should retain their non-zero weights.
    neighbourhood_radius = 80   : int       # Radius in um of the area around the ROI where matches are considered. Default of 80 translates to 50 pixels at 1x zoom
    nearby_neuron_cap = 15      : int       # Maximum nearest neurons found by KDTree nearest-neighbor-analysis
    fov_shift_patches = 8       : tinyint   # Number of patches squared into which the FOV will be split for shift estimation. E.g. with 8, the FOV will be split into 8x8=64 patches.
    -> CellMatchingClassifier
    match_param_description  : varchar(512) # Short description of the background and effect of the parameter set
    """


@schema
class CellMatchingClassifier(dj.Lookup):
    definition = """ # Table to store decision tree classifier models for cell matching GUI.
    classifier_id   : int           # Index of classifier version
    ------
    model_path                      : varchar(256)  # Path to classifier model, relative to Wahl folder server
    description                     : varchar(256)  # Description of the model
    n_cells                         : smallint      # Number of inputs for the classifier
    n_features                      : tinyint       # Number of features per cell
    """
    contents = [
        [0, 'CellMatching\\models\\model0.pkl', 'Original model trained by Anna Schmidt-Rohr', 3, 12]
    ]


@schema
class FieldOfViewShift(dj.Manual):
    definition = """ # Piecewise FOV shift between two sessions. Used to correct CoM coordinates. Manual table instead 
    # of Computed, because not every Session needs a shift, it is only computed once it is queried via the GUI. 
    -> common_img.QualityControl
    -> CellMatchingParameter
    matched_session : varchar(64)   # Identifier for the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid
    ------
    shifts         : longblob       # 3D array with shape (n_dims, x, y), holding pixel-wise shifts for x (shifts[0]) and y (shifts[1]) coordinates.
    """

    def make(self, key: dict) -> None:
        """
        Compute FOV-shift map between the queried reference and a target image.
        Mean intensity images (avg_image) are used instead of local correlation images because they are a better
        representation of the FOV anatomy, more stable over time, and are less influenced by  activity patterns. Images
        are split into patches, and the shift is calculated for each patch separately with phase correlation. The
        resulting shift map is scaled up and missing values interpolated to the original FOV size to get an estimated
        shift value for each pixel.

        Args:
            key: Primary keys of the reference session in common_img.QualityControl(),
                    ID of the CellMatchingParameter() entry,
                    and identifier string of the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid.
                    It is assumed that the reference and matched sessions come from the same individual mouse.
        """

        from skimage import registration
        from scipy import ndimage

        # Fetch reference FOV, parameter set, and extract the primary keys of the matched session from the ID string
        match_keys = key['matched_session'].split('_')
        match_key = dict(username=key['username'], mouse_id=key['mouse_id'], day=match_keys[0],
                         session_num=int(match_keys[1]), motion_id=int(match_keys[2]))

        print_dict = dict(username=key['username'], mouse_id=key['mouse_id'], day=key['day'],
                          session_num=key['session_num'], motion_id=key['motion_id'])
        print(f"Computing FOV shift between sessions\n\t{print_dict} and \n\t{match_key}")

        fov_ref = (common_img.QualityControl & key).fetch1('avg_image')
        fov_match = (common_img.QualityControl & match_key).fetch1('avg_image')
        params = (CellMatchingParameter & key).fetch1()

        # Calculate pixel size of each patch
        img_dim = fov_ref.shape
        patch_size = int(img_dim[0] / params['fov_shift_patches'])

        # Shift maps are a 2D matrix of shape (n_patch, n_patch), with the phase correlation of each patch
        shift_map = np.zeros((2, params['fov_shift_patches'], params['fov_shift_patches']))
        for row in range(params['fov_shift_patches']):
            for col in range(params['fov_shift_patches']):
                # Get a view of the current patch by slicing rows and columns of the FOVs
                curr_ref_patch = fov_ref[row * patch_size:row * patch_size + patch_size,
                                         col * patch_size:col * patch_size + patch_size]
                curr_tar_patch = fov_match[row * patch_size:row * patch_size + patch_size,
                                           col * patch_size:col * patch_size + patch_size]
                # Perform phase cross correlation to estimate image translation shift for each patch
                patch_shift = registration.phase_cross_correlation(curr_ref_patch, curr_tar_patch, upsample_factor=100,
                                                                   return_error=False)
                shift_map[:, row, col] = patch_shift

        # Use scipy's zoom to upscale single-patch shifts to FOV size and get pixel-wise shifts via spline interpolation
        x_shift_map_big = ndimage.zoom(shift_map[0], patch_size, order=3)   # Zoom X and Y shifts separately
        y_shift_map_big = ndimage.zoom(shift_map[1], patch_size, order=3)
        # Further smoothing (e.g. Gaussian) is not necessary, the interpolation during zooming smoothes harsh borders
        shift_map_big = np.stack((x_shift_map_big, y_shift_map_big))

        # If caiman_id is in the primary keys, remove it, because it is not in QualityControl, this table's parent
        if 'caiman_id' in key:
            del key['caiman_id']

        # Insert shift map into the table
        self.insert1(dict(**key, shifts=shift_map_big))


@schema
class MatchingFeatures(dj.Computed):
    definition = """ # Features of the ROIs that are used for the cell matching algorithm
    -> common_img.Segmentation
    -> CellMatchingParameter
    ------
    match_time = CURRENT_TIMESTAMP  : timestamp
    """

    class ROI(dj.Part):
        definition = """ # Features of neurons that are used for the cell matching algorithm
        -> master
        mask_id                     : int           # ID of the ROI (same as common_img.Segmentation)
        ------
        contour                     : longblob      # (Semi)-binarized spatial contour of the ROI, maximally cropped
        neighbourhood               : longblob      # Local area of the mean intensity template around the neuron
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

        print(f"Computing matching features for ROIs in entry {key}.")

        # Fetch relevant data
        footprints = (common_img.Segmentation.ROI & key).get_rois()
        coms = np.vstack((common_img.Segmentation.ROI & key).fetch('com'))
        template = (common_img.QualityControl & key).fetch1("avg_image")
        params = (CellMatchingParameter & key).fetch1()

        # Convert neighbourhood radius (in microns) to zoom-dependent radius in pixels
        mean_res = np.mean((common_img.CaimanParameter & key).get_parameter_obj(key)['dxy'])
        margin_px = int(np.round(params['neighbourhood_radius'] / mean_res))

        coms_list = [list(com) for com in coms]  # KDTree expects a list of lists as input
        neighbor_tree = spatial.KDTree(coms_list)  # Build kd-tree to query nearest neighbours of any ROI

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
            num_neurons_in_radius = bisect.bisect(distance, 50) - 1  # -1 to not count the ROI itself
            index_in_radius = index[1: max(0, num_neurons_in_radius) + 1]  # start at 1 to not count the ROI itself

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
                quadrant_split[0] += 1  # top-left quadrant
            elif x_ref <= x and y_ref >= y:
                quadrant_split[1] += 1  # top-right quadrant
            elif x_ref >= x and y_ref >= y:
                quadrant_split[2] += 1  # bottom-left quadrant
            else:
                quadrant_split[3] += 1  # bottom-right quadrant
        return quadrant_split


@schema
class MatchedIndex(dj.Manual):
    definition = """ # Matched indices of ROIs in other sessions, created through Cell Matching GUI
    -> MatchingFeatures.ROI
    matched_session : varchar(64)   # Identifier for the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid
    ------
    matched_id      : int           # Mask ID of the same neuron in the matched session
    matched_time = CURRENT_TIMESTAMP  : timestamp
    """

    def helper_insert1(self, key: dict) -> Optional[bool]:
        """
        Helper function that inserts a confirmed neuron match for both sessions and warns if a cell has been tracked
        twice with different reference cells.

        Args:
            key: Primary keys of the current matched cell entry
        """

        popup_msg = '\nPress "OK" to overwrite entry, or "Cancel" to keep existing entry.'

        dup_keys = key.copy()
        del dup_keys['matched_id']
        overwriting = False

        if len(self & dup_keys) > 0:
            duplicate_entry = (self & dup_keys).fetch1()
            # If the entry already exists, but with another ID, ask the user if it should be overwritten
            if duplicate_entry['matched_id'] != key['matched_id']:
                msg = f'Cell {key["mask_id"]} in session {key["day"]} has a recorded match for session' \
                      f'{duplicate_entry["matched_session"][:11]} with cell {duplicate_entry["matched_id"]}, ' \
                      f'you selected a match with cell {key["matched_id"]}.' + popup_msg
                # This creates a popup window. The two last parameters determine the style, OR'd together. The first
                # gives the window an "OK" and a "Cancel" button, the second draws the window above all other windows
                response = ctypes.windll.user32.MessageBoxW(0, msg, 'Conflicting entry!', 0x00000001 | 0x0004)
                if response == 1:
                    self.update1(key)
                    # print('Would have updated first match:', key)
                    overwriting = True
                else:
                    return False
            # If the same entry already exists, we dont have to check the reverse entry as well.
            else:
                return False
        else:
            # Insert main entry
            self.insert1(key)
            # print('Would have inserted first match:', key)

        # Insert reverse entry (if cell X in session A is the same as cell Y in session B, then Y(B) should also be
        # the same cell as X(A))
        reverse_key = dict(username=key['username'],
                           mouse_id=key['mouse_id'],
                           day=key['matched_session'].split('_')[0],
                           session_num=key['matched_session'].split('_')[1],
                           motion_id=key['matched_session'].split('_')[2],
                           caiman_id=key['matched_session'].split('_')[3],
                           match_param_id=key['match_param_id'],
                           mask_id=key['matched_id'],
                           matched_session=f"{key['day']}_{key['session_num']}_{key['motion_id']}_{key['caiman_id']}",
                           matched_id=key['mask_id'])

        # To check for possible duplicates, we have to find entries with the same matched ID, but different mask ID
        dup_key = reverse_key.copy()
        del dup_key['mask_id']

        # Filter out no-match sessions
        if reverse_key['mask_id'] != -1:
            if len(self & dup_key) == 1:
                # If the entry already exists, but with another ID, ask the user if it should be overwritten
                if (self & dup_key).fetch1('mask_id') != reverse_key['mask_id']:
                    # If the first entry was overwritten already, we dont have to ask again and just overwrite this one as well
                    if overwriting:
                        self.update1(reverse_key)
                        # print('Would have updated reverse match:', reverse_key)
                    else:
                        msg = f"Cell {reverse_key['mask_id']} in session {reverse_key['day']} is already matched to " \
                              f"Cell {(self & reverse_key).fetch1('matched_id')} in session {reverse_key['matched_session']}." \
                              f"You matched it to Cell ID {reverse_key['matched_id']} instead." + popup_msg
                        response = ctypes.windll.user32.MessageBoxW(0, msg, 'Conflicting entry!', 0x00000001 | 0x0004)
                        if response == 1:
                            self.update1(reverse_key)
                            # print('Would have updated reverse match:', reverse_key)
                        else:
                            return False
                else:
                    print('Match already in database, insert skipped.')
                    return False
            else:
                self.insert1(reverse_key)
                # print('Would have inserted reverse match:', reverse_key)

    def remove_match(self, key: dict) -> None:
        """
        Remove matches of one cell and its reverse matches in all sessions.

        Args:
            key: Primary keys, need to identify the reference cell
        """

        day = (self & key).fetch('day')
        mask_id = (self & key).fetch('mask_id')

        if len(np.unique(day)) > 1:
            raise KeyError('Provided key has to specify a single session.')
        elif len(np.unique(mask_id)) > 1:
            raise KeyError('Provided key has to specify a single ROI.')
        else:
            sessions = self & f'day="{day[0]}"' & f'mask_id={mask_id[0]}'
            rev_sessions = self & f'matched_id={mask_id[0]}'

            print('These matches will be deleted:\n', sessions)
            print('These reverse matches will be deleted:\n', rev_sessions)
            response = input('Confirm? (y/n)')
            if response == 'y':
                sessions.delete()
                rev_sessions.delete()
            else:
                print('Aborted.')
                return

