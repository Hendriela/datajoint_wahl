#!/bin/sh
# This script should run automatically on bootup and start the Datajoint MySQL database.
cd /db/mysql-docker
docker-compose up -d
python3 /home/hheise/datajoint_wahl/datajoint_wahl/util/notify_server_restart.py