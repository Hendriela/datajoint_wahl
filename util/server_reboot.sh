#!/bin/bash
# This script should run automatically on bootup and setup several things

# Set the terminal program to fix a bug with Nano on Ubuntu. If not sufficient, add this line to ~/.bashrc
export TERM=xterm-color

# start the Datajoint MySQL database.
cd /db/mysql-docker
docker-compose up -d
python3 /home/hheise/datajoint_wahl/util/notify_server_restart.py