# Apache RAT License Check

This is a nightly test that runs the Apache Tool RAT to check the License Headers on all source files
 
### The .rat-excludes file
This file lists all the files, directories and file formats that are excluded from license checks for various reasons.
If you think something is wrong, feel free to change!

### Nightly test script for license check
Coming soon...

### How to run the RAT check locally
The following commands can be used to run a Apache RAT check locally - 

```
#install maven
sudo apt-get install maven -y #>/dev/null

#install svn
sudo apt-get install subversion -y #>/dev/null

#download RAT
svn co http://svn.apache.org/repos/asf/creadur/rat/trunk/ #>/dev/null

#cd into correct directory
cd trunk

#install step
mvn install #>/dev/null

#If build success:
cd apache-rat/target

#run Apache RAT check on the src
java -jar apache-rat-0.13-SNAPSHOT.jar -E <path-to-.rat-excludes-file> -d <path-to-mxnet-source>
```
