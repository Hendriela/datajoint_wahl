#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 25/05/2022 18:47
@author: hheise

Scripts to interact with the Allen Brain Atlas API
"""

import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from typing import Optional


def download_annotated_atlas(resolution: int, nth_slice: int, save_to: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Wrapper function for the Allen API to download the annotated reference atlas.

    Args:
        resolution  : Resolution of the atlas in [um/voxel]. Possible values are 10, 25, 50 and 100.
        nth_slice   : Which slices to keep. e.g. a value of 4 keeps every 4th slice. This is to fit the resolution to the number of slices, e.g. the interactive online version only shows slices every 100 um.
        save_to     : Optional absolute path of the saved file. If provided, the atlas is saved as a numpy array instead of being returned.

    Returns:
        If save_to is not given (default), returns a np.ndarray with dimensions (A-P, D-V, M-L). Size of the array
            depends on the given resolution and nth_slice.
        If save_to is given, the array is stored at that location and nothing is returned
    """
    # Location of the current annotation atlas of CCFv3 on the Allen servers. Update this if a new CCF is released.
    reference_space_key = 'annotation/ccf_2017'

    # Set up the instance that deals with the raw NRRP data coming from the Allen Institute servers
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    # Download the annotation atlas
    annotation, meta = rspc.get_annotation_volume()

    # Only keep the n_th slice
    annot_down = annotation[::nth_slice]
    # The first slice is empty, so replace it with the second, which actually holds the first brain structures
    annot_down[0] = annotation[1]

    # Save atlas to disk or return it
    if save_to is not None:
        np.save(save_to, annot_down)
    else:
        return annot_down
