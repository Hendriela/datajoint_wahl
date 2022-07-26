#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 23/03/2022 20:54
@author: Hendrik

Functions to import already processed
"""

from glob import glob
import os
import pickle
import numpy as np

from schema import common_img, common_exp
from caiman.source_extraction.cnmf import cnmf

def load_pcf(root, fname=None):
    if fname is not None:
        pcf_path = glob(os.path.join(root, fname+'.pickle'))
        if len(pcf_path) < 1:
            pcf_path = glob(os.path.join(root, fname))
            if len(pcf_path) < 1:
                raise FileNotFoundError(f'No pcf file found in {os.path.join(root, fname)}.')
    else:
        pcf_path = glob(root + r'\\pcf_results*')
        if len(pcf_path) < 1:
            raise FileNotFoundError(f'No pcf file found in {root}.')
        elif len(pcf_path) > 1:
            pcf_path = glob(root + r'\\pcf_results_manual.pickle')
            if len(pcf_path) < 1:
                pcf_path = glob(root + r'\\pcf_results.pickle')
                if len(pcf_path) < 1:
                    pcf_path = glob(root + r'\\pcf_results')
                elif len(pcf_path) > 1:
                    raise FileNotFoundError(f'More than one pcf file found in {root}.')
    print(f'Loading file {pcf_path[0]}...')
    with open(pcf_path[0], 'rb') as file:
        obj = pickle.load(file)

    return obj

def pipeline_with_imported_params(username, mouse_ids):

    mice = [{'username': username, 'mouse_id': mouse_id} for mouse_id in mouse_ids]

    for mouse in mice:
        schedule = ((common_img.ScanInfo & mouse) - common_img.MotionCorrection).fetch('KEY', as_dict=True)

        for session in schedule:
            # Try to load the cnm parameters from an HDF5 or pickle file
            sess_dir = (common_exp.Session & session).get_absolute_path()

            try:
                cnm_file = glob(os.path.join(sess_dir, 'cnm_*.hdf5'))
                params = cnmf.load_CNMF(cnm_file[0]).params
            except IndexError:
                params = load_pcf(sess_dir).cnmf.params

            # Extract parameters and construct entry dicts
            zoom = {'zoom': (common_img.ScanInfo & session).fetch1('zoom')}
            fov = ((common_img.FieldOfViewSize & zoom).fetch1('x'), (common_img.FieldOfViewSize & zoom).fetch1('y'))
            dxy = (fov[0] / (common_img.ScanInfo & session).fetch1('pixel_per_line'),
                   fov[1] / (common_img.ScanInfo & session).fetch1('nr_lines'))
            border_nan = 0 if params.motion['border_nan'] == 'copy' else 1

            motion_params = dict(max_shift=int(np.round(params.motion['max_shifts'][0]*np.mean(dxy))),
                                 stride_mc=int(np.round(params.motion['strides'][0]*np.mean(dxy))),
                                 overlap_mc=params.motion['overlaps'][0],
                                 pw_rigid=int(params.motion['pw_rigid']),
                                 max_dev_rigid=params.motion['max_deviation_rigid'],
                                 border_nan=border_nan,
                                 n_iter_rig=params.motion['niter_rig'],
                                 nonneg_movie=int(params.motion['nonneg_movie']))

            if len(common_img.MotionParameter & motion_params) == 0:
                # If the current set of parameters are not in the database, enter them with an automatic description
                mot_id = np.max(common_img.MotionParameter().fetch('motion_id'))+1
                entry = dict(motion_id=mot_id,
                             motion_shortname=f'M{session["mouse_id"]}, {session["day"]}',
                             motion_description=f'Parameter set used in early pipeline. Automatically adopted upon '
                                                f'import into DataJoint. First encountered in session '
                                                f'M{session["mouse_id"]}_{session["day"]}',
                             **motion_params)
                common_img.MotionParameter().insert1(entry)
                print(f'\tMotion parameters are unique, new entry added (motion_id={mot_id}).')
            else:
                mot_id = (common_img.MotionParameter & motion_params).fetch1('motion_id')
                print(f'\tMotion parameters already in database (motion_id={mot_id}).')

            caiman_params = dict(username=username,
                                 mouse_id=session['mouse_id'],
                                 p=params.preprocess['p'],
                                 nb=params.init['nb'],
                                 merge_thr=params.merging['merge_thr'],
                                 stride_cnmf=params.patch['stride'],
                                 k=params.init['K'],
                                 g_sig=params.init['gSig'][0],
                                 method_init=params.init['method_init'],
                                 ssub=params.init['ssub'],
                                 tsub=params.init['tsub'],
                                 snr_lowest=float(params.quality['SNR_lowest']),
                                 snr_thr=float(params.quality['min_SNR']),
                                 rval_lowest=float(params.quality['rval_lowest']),
                                 rval_thr=float(params.quality['rval_thr']),
                                 cnn_lowest=float(params.quality['cnn_lowest']),
                                 cnn_thr=float(params.quality['min_cnn_thr']))

            if params.patch['rf'] is not None:
                caiman_params['rf'] = params.patch['rf']

            # Todo: figure out why filtering CaimanParameter with caiman_params does not work? (something with rval_thr, cnn_lowest and cnn_thresh that does not work when comparing)
            # if len(common_img.CaimanParameter & caiman_params) == 0:

            existing_params = (common_img.CaimanParameter & f'username="{username}"' &
                               f'mouse_id={session["mouse_id"]}').fetch(as_dict=True)
            # Check if the current parameter set already exists (is a subset of a CaimanParameter entry)
            is_subset = [caiman_params.items() <= x.items() for x in existing_params]

            if len(existing_params) == 0 or not any(is_subset):
                # If the current set of parameters are not in the database, enter them with an automatic description
                cnm_ids = (common_img.CaimanParameter & f'username="{username}"' & f'mouse_id={session["mouse_id"]}').fetch('caiman_id')
                cnm_id = 0 if len(cnm_ids) == 0 else np.max(cnm_ids)+1
                entry = dict(caiman_id=cnm_id, **caiman_params)
                common_img.CaimanParameter().insert1(entry)
                print(f'\tCaiman parameters are unique, new entry added (caiman_id={cnm_id}).')

            elif sum(is_subset) == 1:
                # If there is one matching parameter set, take its ID as the caiman_id for this session
                cnm_id = existing_params[is_subset.index(True)]['caiman_id']
                print(f'\tCaiman parameters already in database (caiman_id={cnm_id}).')

            elif sum(is_subset) > 1:
                raise ValueError(f'More than one set found that matches all Caiman parameters for session {session}')

            else:
                raise NotImplementedError(f'This should not occur: Session {session}')

            # If the current parameter set is not the standard ID, enter it into CaimanParameterSession which keeps
            # track of which parameter set is for which session
            if cnm_id != 0:
                common_img.CaimanParameterSession().insert1((dict(**session, caiman_id=cnm_id)), skip_duplicates=True)

            # Now we can fill the entries for MotionCorrection, QualityControl and Segmentation via chain-piping
            session['motion_id'] = mot_id
            session['caiman_id'] = cnm_id

            common_img.MotionCorrection().populate(session, make_kwargs=dict(chain_pipeline=True, caiman_id=cnm_id))
