<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Source Download

These source archives are generated from tagged releases. Updates and patches will not have been applied. For any updates refer to the corresponding branches in the [GitHub repository](https://github.com/apache/incubator-mxnet). Choose your flavor of download from the following links:

| Version | Source                                                                                                      | PGP                                                                                                             | SHA                                                                                                                |
|---------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| 1.3.1   | [Download](https://www.apache.org/dyn/closer.cgi/incubator/mxnet/1.3.1/apache-mxnet-src-1.3.1-incubating.tar.gz)   | [Download](https://apache.org/dist/incubator/mxnet/1.3.1/apache-mxnet-src-1.3.1-incubating.tar.gz.asc)    | [Download](https://apache.org/dist/incubator/mxnet/1.3.1/apache-mxnet-src-1.3.1-incubating.tar.gz.sha512)      |
| 1.3.0   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.3.0/apache-mxnet-src-1.3.0-incubating.tar.gz)   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.3.0/apache-mxnet-src-1.3.0-incubating.tar.gz.asc)    | [Download](https://archive.apache.org/dist/incubator/mxnet/1.3.0/apache-mxnet-src-1.3.0-incubating.tar.gz.sha512)      |
| 1.2.1   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.2.1/apache-mxnet-src-1.2.1-incubating.tar.gz)   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.2.1/apache-mxnet-src-1.2.1-incubating.tar.gz.asc)    | [Download](https://archive.apache.org/dist/incubator/mxnet/1.2.1/apache-mxnet-src-1.2.1-incubating.tar.gz.sha512)      |
| 1.2.0   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.2.0/apache-mxnet-src-1.2.0-incubating.tar.gz)   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.2.0/apache-mxnet-src-1.2.0-incubating.tar.gz.asc)    | [Download](https://archive.apache.org/dist/incubator/mxnet/1.2.0/apache-mxnet-src-1.2.0-incubating.tar.gz.sha512)      |
| 1.1.0   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.1.0/apache-mxnet-src-1.1.0-incubating.tar.gz)      | [Download](https://archive.apache.org/dist/incubator/mxnet/1.1.0/apache-mxnet-src-1.1.0-incubating.tar.gz.asc)      | [Download](https://archive.apache.org/dist/incubator/mxnet/1.1.0/apache-mxnet-src-1.1.0-incubating.tar.gz.sha512)     |
| 1.0.0   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.0.0/apache-mxnet-src-1.0.0-incubating.tar.gz)   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.0.0/apache-mxnet-src-1.0.0-incubating.tar.gz.asc)   | [Download](https://archive.apache.org/dist/incubator/mxnet/1.0.0/apache-mxnet-src-1.0.0-incubating.tar.gz.sha512)   |
| 0.12.1  | [Download](https://archive.apache.org/dist/incubator/mxnet/0.12.1/apache-mxnet-src-0.12.1-incubating.tar.gz) | [Download](https://archive.apache.org/dist/incubator/mxnet/0.12.1/apache-mxnet-src-0.12.1-incubating.tar.gz.asc) | [Download](https://archive.apache.org/dist/incubator/mxnet/0.12.1/apache-mxnet-src-0.12.1-incubating.tar.gz.sha512) |
| 0.12.0  | [Download](https://archive.apache.org/dist/incubator/mxnet/0.12.0/apache-mxnet-src-0.12.0-incubating.tar.gz) | [Download](https://archive.apache.org/dist/incubator/mxnet/0.12.0/apache-mxnet-src-0.12.0-incubating.tar.gz.asc) | [Download](https://archive.apache.org/dist/incubator/mxnet/0.12.0/apache-mxnet-src-0.12.0-incubating.tar.gz.sha512) |
| 0.11.0  | [Download](https://archive.apache.org/dist/incubator/mxnet/0.11.0/apache-mxnet-src-0.11.0-incubating.tar.gz) | [Download](https://archive.apache.org/dist/incubator/mxnet/0.11.0/apache-mxnet-src-0.11.0-incubating.tar.gz.asc) | [Download](https://archive.apache.org/dist/incubator/mxnet/0.11.0/apache-mxnet-src-0.11.0-incubating.tar.gz.sha512) |

## Verify the Integrity of the Files
It is essential that you verify the integrity of the downloaded file using the PGP signature (.asc file) or a hash (.md5 or .sha* file). Please read [Verifying Apache Software Foundation Releases](https://www.apache.org/info/verification.html) for more information on why you should verify our releases.

The PGP signature can be verified using PGP or GPG. First download the KEYS as well as the .asc signature file for the relevant distribution. Make sure you get these files from the main distribution site, rather than from a mirror. Then verify the signatures using one of the following alternatives:

```bash
% gpg --import KEYS
% gpg --verify downloaded_file.asc downloaded_file
```

```bash
% pgpk -a KEYS
% pgpv downloaded_file.asc
```

```bash
% pgp -ka KEYS
% pgp downloaded_file.asc
```

Alternatively, you can verify the hash on the file.

Hashes can be calculated using GPG:

```bash
% gpg --print-md SHA1 downloaded_file
```

The output should be compared with the contents of the SHA1 file. Similarly for other hashes (SHA256 MD5 etc) which may be provided.

Windows 7 and later systems should all now have `certUtil`:

```bash
% certUtil -hashfile pathToFileToCheck
```

HashAlgorithm choices: MD2 MD4 MD5 SHA1 SHA256 SHA384 SHA512

Unix-like systems (and macOS) will have a utility called `md5`, `md5sum` or `shasum`.
