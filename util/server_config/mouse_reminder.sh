#!/bin/bash
# This script runs weekly and reminds DataJoint investigators about mice without a recorded weight
echo "$(date): Start sending out mouse reminders"
/home/hheise/datajoint_wahl/bin/python3 /home/hheise/datajoint_wahl/util/server_config/mouse_reminder.py
echo "Done!"