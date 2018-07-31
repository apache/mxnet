# Broken link checker test

This folder contains the scripts that are required to run the nightly job of checking the broken links. The job also checks whether the link that were published before are still accessible.
 
## JenkinsfileForBLC
This is configuration file for jenkins job.

## Details
The `broken_link_checker.sh` is a top level script that invokes the `test_broken_links.py` and `check_regression.sh` scripts.
The `test_broken_links.py` invokes broken link checker tool (blc) from nodeJs and reports the list of URLs that are not accessible.
The `check_regression.sh` scripts downloads the file `url_list.txt` that contains links that are publicly accessible from s3 bucket
The scripts merges this list with the output of `test_broken_links.py` and checks whether all those links are accessible using 'curl' command.
The updated `url_list.txt` is uploaded to s3 bucket.
