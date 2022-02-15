#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 05/10/2021 11:24
@author: hheise

A few small helper functions.
"""

from typing import List
import re
import datajoint as dj
import os
from datetime import datetime
import pickle


def alphanumerical_sort(x: List[str]) -> List[str]:
    """
    Sorts a list of strings alpha-numerically, with proper interpretation of numbers in the string. Usually used to sort
    file names to ensure files are processed in the correct order (if file name numbers are not zero-padded).
    Elements are first  sorted by characters. If characters are the same, elements are sorted by numbers in the string.
    Consecutive digits are treated as one number and sorted accordingly (e.g. "text_11" will be sorted behind "text_2").
    Leading Zeros are ignored.

    Args:
        x:  List to be sorted. Elements are usually file names.

    Returns:
        Sorted list.
    """
    x_sort = x.copy()

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    x_sort.sort(key=natural_keys)
    return x_sort


def extract_documentation(table: dj.Table, return_only_pks=True):
    """ USELESS FUNCTION FOR NOW, BUT MAY BE USEFUL LATER TO IMPLEMENT SUPPORT FOR MULTILINE COMMENTS!!

    Function to extract full documentation from a table. Probably easier to use DataJoint's table.heading, although
    that attribute does not support multiline comments..."""

    def fuse_multiline_comments(lines):

        parsed_lines = []

        for line in lines:
            parsed = line.split("#")
            # If the line has text on both sides of the hashtag, its a normal attribute and can be added to the list
            if len(parsed[0].strip()) > 0 and len(parsed) > 1:
                parsed_lines.append(line[:])
            # If the line has a hashtag, but nothing on the left side of it, its a multiline comment, and the last entry
            # of the list should be extended
            elif len(parsed[0].strip()) == 0 and len(parsed) > 1:
                parsed_lines[-1] += " " + parsed[1].strip()
            # If the line has no hashtag, but text, it might be a foreign key, which we have to keep
            elif (len(parsed[0].strip()) > 0) and (len(parsed[0].strip()) > 1) and ("->" in parsed[0]):
                parsed_lines.append(parsed[0].strip())
            else:
                pass

        return parsed_lines

    # Table.heading returns all primary keys, also foreign keys. However, multiline comments are NOT SUPPORTED
    heading = table.heading

    # Fetch complete definition string
    description = table.definition

    # Separate table description and primary keys from secondary keys
    split_description = description.split("---")

    # Descriptions might include more than three "-" to separate primary from secondary keys, this has to be cleaned up
    table_doc_pk = split_description[0].strip()  # The table docs and primary keys are always the first element
    sec_attributes = split_description[-1].strip("-").strip()  # The last element might include trailing dashes

    # Separate table description from primary keys
    table_doc, *primary_keys = table_doc_pk.split("\n")

    # Remove commenting hashtag and whitespaces. Table description is done.
    table_doc = table_doc.split("#")[-1].strip()

    # Parse all primary keys
    primary_keys = fuse_multiline_comments(primary_keys)  # After this, every element is a primary key (or foreign key)

    pks = {}
    for pk in primary_keys:
        # Normal primary key
        if ":" in pk:
            pks[pk.split(":")[0].strip()] = dict(dtype=pk.split(":")[1].split("#")[0].strip(),
                                                 comment=pk.split(":")[1].split("#")[1].strip())
        # Foreign key
        elif "->" in pk:
            # Todo: if PK (or attr) is inherited, call extract_documentation recursively on that table
            pass
            print("Foreign primary key: ", out)

    if return_only_pks:
        return (pks,)
    else:
        # Separate secondary attributes
        sec_attributes = sec_attributes.split("\n")
        sec_attributes = fuse_multiline_comments(sec_attributes)

        # Parse all secondary attributes
        attr = {}
        for att in sec_attributes:
            # Normal attribute
            if ":" in att:
                attr_name = att.split(":")[0].strip()
                if "=" in attr_name:  # Attribute has a default value
                    attr[attr_name.split("=")[0].strip()] = dict(default_value=attr_name.split("=")[1].strip(),
                                                                 dtype=att.split(":")[1].split("#")[0].strip(),
                                                                 comment=att.split(":")[1].split("#")[1].strip())
                else:
                    attr[attr_name] = dict(dtype=att.split(":")[1].split("#")[0].strip(),
                                           comment=att.split(":")[1].split("#")[1].strip())
            # Foreign key
            elif "->" in att:
                print("Resolving foreign key for: ", att)
                pass

        return (table_doc, pks, attr)


def backup_manual_data(backup_dir='DataJoint\\backups'):
    # Import all schemas for read access
    import login
    login.root_connect()
    from schema import common_mice, common_exp, common_img, common_match, common_dlc, hheise_behav, hheise_placecell, \
        mpanze_behav, mpanze_mapping, mpanze_widefield

    schemas = {'common_mice': ['Mouse', 'Weight', 'PainManagement', 'Sacrificed', 'Surgery', 'Injection'],
               'common_exp': ['Session', 'Anesthesia', 'Setup', 'Task'],
               'common_img': ['Scan', 'RawImagingFile', 'MotionParameter', 'CaimanParameter', 'Microscope', 'Laser', 'Layer', 'BrainRegion', 'FieldOfViewSize', 'CaIndicator'],
               'common_dlc': ['Video', 'RawVideoFile', 'FrameCountVideoTimeFile', 'CameraPosition', 'FFMPEGParameter',
                              'DLCModel', 'MedianFilterParameter', 'InterpolationParameter'],
               'common_match': ['CellMatchingParameter', 'MatchedIndex'],
               'hheise_behav': ['BatchData'],
               'hheise_placecell': ['PlaceCellParameter'],
               'mpanze_behav': ['StrokeDate'],
               'mpanze_mapping': ['MappingSession', 'RawSynchronisationFile', 'RawParameterFile'],
               'mpanze_widefield': ['Scan', 'ScanInfo', 'RawImagingFile', 'ReferenceImage', 'AffineRegistration']}

    backup = {}
    for schema in schemas:
        backup[schema] = {}
        for table in schemas[schema]:
            backup[schema][table] = eval(f'{schema}.{table}().fetch(as_dict=True)')

    backup_path = os.path.join(login.get_neurophys_wahl_directory(), backup_dir)
    curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(os.path.join(backup_path, f'backup_{curr_time}.pickle'), 'wb') as handle:
        pickle.dump(backup, handle)
