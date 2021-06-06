# Datajoint server for the Wahl Group

This repository contains the source code for the Datajoint server maintained by the Wahl group.

## Re-starting the MySQL database

If the database got shut down (e.g. if the server was turned off), it has to be re-started with these steps.

<ol>
  <li>Ensure that the physical machine is running. Connect to it by typing this into a terminal and entering the password:
        <pre>ssh 130.60.53.47 -l hheise</pre> </li>
  <li>Navigate to the docker directory:
        <pre>cd ../../db/mysql-docker</pre></li>
  <li>Start the container:
        <pre>sudo docker-compose up -d</pre></li>
  <li>Now you should see <code>Starting mysql-docker_db_1 ... done</code> and the server should be running. Disconnect from the machine:
        <pre>logout</pre></li>        
</ol>

 Consult the <a href="https://github.com/datajoint/mysql-docker">source code</a> of the mysql-docker if more problems arise.