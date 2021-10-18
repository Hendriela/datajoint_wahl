#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/10/2021 18:21
@author: hheise

Script that is sent out by the Ubuntu server upon reboot.
"""
import smtplib
import ssl
from datetime import datetime

port = 465
context = ssl.create_default_context()
sender = 'datajoint.wahl@gmail.com'
now = datetime.strftime(datetime.now(), "%Y-%m-%d, %H:%M:%S")
message = """\
Subject: Datajoint server restarted

This is an automated notice that the Ubuntu machine rebooted and the MySQL server has been automatically restarted.
Time: {}""".format(now)

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(sender, 'password exchanged manually on server')
    server.sendmail(sender, 'heiser@hifo.uzh.ch', message)
