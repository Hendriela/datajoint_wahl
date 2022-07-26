#!/bin/bash
# This script should run automatically on bootup and setup several things

# Set the terminal program to fix a bug with Nano on Ubuntu. If not sufficient, add this line to ~/.bashrc
echo "$(date): Machine rebooted"
export TERM=xterm-color

# start the Datajoint MySQL database.
echo "Restarting MySQL Docker image..."
cd /db/mysql-docker
docker-compose up -d
echo "Sending Email notification..."
python3 /home/hheise/datajoint_wahl/util/server_config/notify_server_restart.py
echo "Done!"