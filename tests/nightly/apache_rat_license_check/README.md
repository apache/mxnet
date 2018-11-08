# Apache RAT License Check

This is a nightly test that runs the Apache Tool RAT to check the License Headers on all source files
 
### The .rat-excludes file
This file lists all the files, directories and file formats that are excluded from license checks for various reasons.
If you think something is wrong, feel free to change!

### Nightly test script for license check
The license check script called by the Jenkinsfile

### How to run the RAT check locally
The following commands can be used to run a Apache RAT check locally - 

Docker based 1-click-method:
```
ci/build.py -p ubuntu_rat nightly_test_rat_check
```

Manual method:
```
#install maven
sudo apt-get install maven -y #>/dev/null

#install svn
sudo apt-get install subversion -y #>/dev/null

#download RAT 0.12 version
svn co http://svn.apache.org/repos/asf/creadur/rat/tags/apache-rat-project-0.12-RC3/ #>/dev/null

#cd into correct directory
cd trunk

#install step
mvn install #>/dev/null

#If build success:
cd apache-rat/target

#run Apache RAT check on the src
java -jar apache-rat-0.12.jar -E <path-to-.rat-excludes-file> -d <path-to-mxnet-source>
```
