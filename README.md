# Datajoint server for the Wahl Group

This repository contains the source code for the Datajoint server maintained by the Wahl group.

## Necessary source code edits

To optimise our pipeline, a few changes to the Caiman and DataJoint source code were introduced. Until these changes are patched in, they have to be done manually.
The scripts that have to be changed are in the <code>edit_source</code> directory. The location of the original script that has to be replaced is at the top of each adapted script. The exact path is user-specific and depends a.o. on the name of your environments and drives. Replace the content of the original script with our adapted version before running the pipeline.

## Re-starting the MySQL database

If the database is unreachable, but the server machine is running (SSH works), it has to be re-started. The database should automatically restart upon reboot of the machine, so restarting the machine should work as well (it takes a minute after the machine is restarted for the database to go live again). However, if it does not work, you can manually restart the database with these steps:

<ol>
  <li>Ensure that the physical machine is running. Connect to it by typing this into a terminal and entering the password:
        <pre>ssh 130.60.53.48 -l hheise</pre> </li>
  <li>Navigate to the docker directory:
        <pre>cd /db/mysql-docker</pre></li>
  <li>Start the container:
        <pre>sudo docker-compose up -d</pre>
      The computer might potentially ask you for the admin password again.</li>
  <li>Now you should see <code>Starting mysql-docker_db_1 ... done</code> and the server should be running. Disconnect from the machine:
        <pre>logout</pre></li>        
</ol>

 Consult the <a href="https://github.com/datajoint/mysql-docker">source code</a> of the mysql-docker if more problems arise.


## Access the MySQL database

To access the MySQL database directly (for maintenance, user creation etc.), connect to the server via SSH and type: <pre>mysql -h 127.0.0.1 -u YourAccountName -p</pre> and enter your password. Then the <code>mysql></code> prompt should appear and you can start using MySQL commands to interact with the database.

### User management
Consult the <a href="https://docs.datajoint.io/matlab/v3.4/admin/3-accounts.html">DataJoint Documentation</a> for creating new users and granting privileges. To create new users or change privileges, you have to use the <code>root</code> account to log into the database.

### Some useful MySQL commands

Since MySQL is a very common database language, it is usually easiest to google whatever you want to do with the database, and find the exact syntax you are looking for. However, for convenience, here are some commands for the most common procedures:

#### Remove old or corrupted schemas
Sometimes things go wrong with DataJoint, and it cannot access schemas anymore. This happens e.g. if the respective python module (its creation .py file) is not accessible anymore, and can create reference and integrity problems. To drop such a corrupted schema <code>old_schema</code>, issue the SQL query <pre>mysql> DROP SCHEMA old_schema;</pre> <u><b>CAUTION:</b></u> This action deletes an ENTIRE schema with all its tables and data, without any further check, safety net or rollback possibility, and is <b><u>IRREVERSIBLE!</u></b> Please use this command with the utmost care. As with user management, you have to log in to MySQL with the <code>root</code> account to drop schemas and tables.

#### Clean up a schema
Schemas and its tables are created and managed by DataJoint, but exist in a folder format on the server. These two things should be the same, but it can happen that tables were created, but its reference lost, creating orphan folders on the server that are inaccessible through DataJoint.
This can be avoided by always calling <code>my_table.drop()</code> when changing the table type or definition before reloading/reimporting the schema (the <code>my_schema.py</code> script). This removes the table as well as the folder from the server, keeping it clean. If you change the table and reload the schema, the old table might remain and be orphaned.
If this happens, we can remove it directly in the MySQL database:

To interact with tables of a specific schema, we have to tell the database which schema we want to use: <pre>mysql> USE my_schema;</pre>
We then can list all tables of this schema with <pre>mysql> SHOW TABLES;</pre>
In the upcoming list of tables, the different DataJoint table types are marked by different prefixes before their actual name: <code>Manual</code> tables have no prefix, <code>Lookup</code> tables have a <code>#</code>, <code>Imported</code> tables a single <code>\_</code>, <code>Computed</code> tables a double <code>\_\_</code>, and <code>Part</code> tables are separated from their main table with another double <code>\_\_</code>.
You can delete a table with <pre>mysql> DROP TABLE \`my_table\`;</pre>
Pay attention that the table name should be surrounded (escaped) by BACKTICKS <code>`</code>, not single or double apostrophes!
