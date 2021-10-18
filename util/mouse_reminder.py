#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/10/2021 10:22
@author: hheise

Automatically send email with mice that have still to be weighed this week.
"""

import smtplib
import ssl
import keyring
from datetime import datetime
import numpy as np
from typing import Optional, List

# Custom imports
import login
from schema import common_mice


def get_last_weighing_date(mouse: dict) -> Optional[datetime]:
    """
    Looks up the last date of a common_mice.Weight() entry for a specific mouse.
    Returns None if mouse has no weight entries.

    Args:
        mouse: (Primary) keys of the queried mouse.

    Returns:
        A datetime object of the last weight entry, None if mouse does not have any entries.
    """
    try:
        last_date = np.max((common_mice.Weight & mouse).fetch('date_of_weight'))
    except ValueError:
        # This happens if a mouse has no weight entries at all
        last_date = None
    return last_date


def get_due_mice(username: str) -> List[dict]:
    """
    Get a list of mice from the queried investigator that have their last weight day at least 7 days ago.

    Args:
        username: Shortname of the queried investigator

    Returns:
        A list of due mice (one entry per mouse), each entry contains a dictionary of relevant data.
    """
    # Query all alive mice of that investigator
    mice = (common_mice.Mouse - common_mice.Sacrificed & f'username="{username}"').fetch()

    # mice = (common_mice.Mouse  & f'mouse_id=95').fetch()
    # Initialize list that stores info of due mice
    due_mice = []

    now = datetime.now().date()

    for mouse in mice:
        # Get the date of the last weight for each mouse
        date = get_last_weighing_date(mouse)
        # If the last weight day is 7 days or more ago, or no weight has been recorded, add relevant data to due_mice
        if date is None:
            due_mice.append({'Last weight day': 'No weight on record!',
                             'Days since last weighing': 'No weight on record!',
                             'Mouse ID': mouse['mouse_id'],
                             'Cage number': mouse['cage_num'],
                             'Date of birth': datetime.strftime(mouse['dob'], '%Y-%m-%d'),
                             'Strain': mouse['strain'],
                             'Additional info': mouse['info']})
        elif (now - date).days >= 7:
            due_mice.append({'Last weight day': datetime.strftime(date, '%Y-%m-%d'),
                             'Days since last weighing': (now - date).days,
                             'Mouse ID': mouse['mouse_id'],
                             'Cage number': mouse['cage_num'],
                             'Date of birth': datetime.strftime(mouse['dob'], '%Y-%m-%d'),
                             'Strain': mouse['strain'],
                             'Additional info': mouse['info']})

    return due_mice


def construct_message(username: str, mouse_list: List[dict]) -> str:
    """
    Construct the email message to the investigator with due mouse info from get_due_mice().

    Args:
        username:   Shortname of the queried investigator
        mouse_list:     Info of due mice, from get_due_mice()

    Returns:
        Message string that can be sent via email
    """

    mouse_data = ""
    for mouse in mouse_list:
        new_mouse = "\n{}".format(mouse)
        mouse_data += new_mouse

    msg = """\
Subject: You have mice that need to be checked and weighed (Automatic Datajoint reminder)
    
Dear {},

this is an automatic reminder that one or more mice associated with you have been checked and weighed
7 or more days ago. You find the relevant info for each due mouse in the list below. Please check and weigh the mice,
then enter the weight into the Datajoint database through the GUI or via commandline. Also update information that
might have changed about the mouse (e.g. different cage number).

Please remember that according to our license 241/2018 you have to check and weigh ALL mice that are "in experiment"
WEEKLY, irrespective of if you performed experiments with these mice this week or not.


The following mice need to be checked and weighed:
    {}

Thank you!

Your friendly neighborhood Datajoint-Bot
    """.format(username, mouse_data)

    return msg


def send_mail(receiver: str, message: str, sender: str = 'datajoint.wahl@gmail.com', port: int = 465) -> None:
    """
    Sends a given plain-text email message from the sender email to the receiver email in a secure SSL context.

    Args:
        receiver:   Email address of the receiver (email address of one of the investigators)
        message:    Plain text string of the email message. The subject is separated by one line break from the body.
        sender:     Email address of the sender (defaults to datajoint.wahl@gmail.com for automatic datajoint emails)
        port:       Port of the SSL server, by default 465.
    """

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender, keyring.get_password('dj_wahl_email', sender))
        server.sendmail(sender, receiver, message)


if __name__ == "__main__":
    try:
        # Connect to the Datajoint server
        login.connect()
    except Exception as ex:
        # If this fails, send the error message as an email to Hendrik's address and terminate the script.
        now = datetime.now()
        msg = """Subject: Automatic mouse reminder email failed.
        
Sending automatic mouse reminder failed at 
{}
with exception
{}: {}.""".format(now, type(ex), ex)

        send_mail('heiser@hifo.uzh.ch', msg)
        raise SystemExit

    # Define sender address to be the datajoint wahl Gmail
    sender_address = 'datajoint.wahl@gmail.com'

    try:
        # Check for due mice
        for investigator in common_mice.Investigator:
            due_mice = get_due_mice(investigator['username'])

            # If there are due mice, construct the email message and send it
            if due_mice:
                msg = construct_message(investigator['username'], due_mice)
                send_mail(investigator['email'], msg)
                
    except Exception as ex:
        # If this fails, send the error message as an email to Hendrik's address and terminate the script.
        now = datetime.now()
        msg = """Subject: Automatic mouse reminder email failed.

Sending automatic mouse reminder failed at 
{}
with exception
{}: {}.""".format(now, type(ex), ex)

        send_mail('heiser@hifo.uzh.ch', msg)



