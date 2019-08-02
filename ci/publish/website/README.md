# Website Deployment

Automatically generated docs for each API are hosted their own folder with the following structure:
* /api/$lang - Example: api/python
* /api/$lang/docs/ - An overview.
* /api/$lang/docs/api/ - the automatically generated API reference
* /api/$lang/docs/guide/ - overview on how to use it and links to important information
* /api/$lang/docs/tutorials/ - overview on the list of tutorials


## Generating Artifacts

You can use the CI scripts to generate artifacts for each language. For example, to generate C++ API docs you can call the following:

```bash
ci/build.py --docker-registry mxnetci --platform ubuntu_cpu_c --docker-build-retries 3 --shm-size 500m /work/runtime_functions.sh build_c_docs
```

This will generate docs for whatever branch you have currently checked out.

Refer to ci/README.md for setup instructions for Docker and docker-python. These are required for running the `build.py` script.

CI stores the artifacts by job run and also has a "latest generated artifacts" link that you can use when a particular docs branch is having issues. You can at least build the website with the latest known version.


## Publishing Artifacts

The artifacts are being hosted on S3 in MXNet's public folder. You must have write access to this bucket to publish new artifacts. You may request access from a committer. Anyone can read from the bucket.

Preview the `publish_artifacts.sh` script and verify the settings. You may want to change the version number, which will affect the bucket location.

## Deploying the Website

The website is deployed automatically several times a day through a Jenkins CI job. This job calls the `deploy.sh` script.

Once the artifacts are available on S3, you can use the `deploy.sh` script to manually deploy the website. This assumes you have the environment variables set for a username and password with write permissions to the `apache/incubator-mxnet-site` repo.
