#!/bin/sh
# Shell script that creates a full backup of the DataJoint database, stores it locally, and then copies it to
# the Neurophysiology-Storage1 Wahl server, which is mounted under /media/
# Arguments used:
# - "--login-path=root" tells mysqldump to use the login credentials for the root account when connecting to the database,
#   avoiding permission errors. For this to work, the root credentials have to be added with the mysql_config_editor
#   (see Linux_server_handling document).
# - "--single-transaction --skip-lock-tables" prevents mysqldump from locking the tables before the backup. Advantage
#   is that tables retain write access during the dump process, which is especially useful for overnight computations
#   that might access the database during the nightly backup.
# - "--all-databases" backs up all databases (schema) of the DataJoint database into a single large file. Single schema
#   can be recovered individually with the "mysql --one-database db1 < datajoint_backup.sql" command.
# - "| gzip > filename" together with the ".gz" in the sql file name compresses the dump file with gzip, which saves
#   disk space and transfer time to the server. When restoring a zipped dump file, you have to add
#   "gunzip < datajoint_backup.sql.gz | mysql..." before the mysql command
echo "starting db backup"
day="$(date +'%A')"
db_backup="datajoint_backup_${day}.sql.gz"
sudo mysqldump --login-path=root --single-transaction --skip-lock-tables --all-databases | gzip >/home/datajoint_backups/${db_backup}
echo "Created backup at ${db_backup}."
sudo cp /home/datajoint_backups/${db_backup} /media/neurophysiology-storage1/Wahl/Datajoint/backups/daily
echo "Copied backup to Wahl server."