# Broken link checker test

This folder contains the scripts that are required to run the nightly job of checking the broken links.
 
## JenkinsfileForBLC
This is configuration file for jenkins job.

## Details
The `broken_link_checker.sh` is a top level script that invokes the `test_broken_links.py` and `check_regression.sh` scripts.
The `test_broken_links.py` invokes broken link checker tool (blc) from nodeJs and reports the list of URLs that are not accessible.
