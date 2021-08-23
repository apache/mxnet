---
layout: page_category
title: MXNet Security Best Practices
category: faq
faq_c: Security
question: How to run MXNet securely?
permalink: /api/faq/security
---
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

# Reporting a security vulnerability
The Apache Software Foundation takes a very active stance in eliminating security problems and denial of service attacks against its products.

We strongly encourage folks to report such problems to our private security mailing list first, before disclosing them in a public forum.

Please note that the security mailing list should only be used for reporting undisclosed security vulnerabilities and managing the process of fixing such vulnerabilities. We cannot accept regular bug reports or other queries at this address. All mail sent to this address that does not relate to an undisclosed security problem in our source code will be ignored.


Questions about:
* if a vulnerability applies to your particular application
* obtaining further information on a published vulnerability
* availability of patches and/or new releases
should be addressed to the users mailing list. Please see the [mailing lists page](/community/contribute#mxnet-dev-communications) for details of how to subscribe.

The private security mailing address is: <a href="mailto:security@apache.org">security@apache.org</a> <i class="far fa-envelope">. Feel free to consult the general [Apache Security guide](http://www.apache.org/security/) for further details about the reporting process.


# MXNet Security Best Practices

MXNet framework has no built-in security protections. It assumes that the MXNet entities involved in model training and inferencing (hosting) are fully trusted. It also assumes that their communications cannot be eavesdropped or tampered with. MXNet consumers shall ensure that the above assumptions are met.

In particular the following threat-vectors exist when training using MXNet:

* When running distributed training using MXNet there is no built-in support for authenticating cluster nodes participating in the training job.
* Data exchange between cluster nodes happens is in plain-text.
* Using `kvstore.set_optimizer` one can use a custom optimizer to combine gradients. This optimizer code is sent to the server nodes as a pickle file. A server does not perform any further validation of the pickle file and simply executes the code trusting the sender (worker).
* Since there is no authentication between nodes, a malicious actor running on the same network can launch a Denial of Service (DoS) attack by sending data that can overwhelm/crash a scheduler or other server nodes.

It is highly recommended that the following best practices be followed when using MXNet:

* Run MXNet with least privilege, i.e. not as root.
* Run MXNet training jobs inside a secure and isolated environment. If you are using a cloud provider like Amazon AWS, running your training job inside a [private VPC](https://aws.amazon.com/vpc/) is a good way to accomplish this. Additionally, configure your network security settings so as to only allow connections that the cluster nodes require.
* Make sure no unauthorized actors have physical or remote access to the nodes participating in MXNet training.
* During training, one can configure MXNet to periodically save model checkpoints. To protect these model checkpoints from unauthorized access, make sure the checkpoints are written out to an encrypted storage volume, and have a provision to delete checkpoints that are no longer needed.
* When sharing trained models, or when receiving trained models from other parties, ensure that model artifacts are authenticated and integrity protected using cryptographic signatures, thus ensuring that the data received comes from trusted sources and has not been maliciously (or accidentally) modified in transit.
* By default, mx.random uses a static and fixed seed value. The random utilities in MXNet should therefore never be used to implement any type of security critical functionality where cryptographically secure pseudorandom number generation is required.

# Deployment Considerations
The following are not MXNet framework specific threats but are applicable to Machine Learning models in general.

* When deploying high-value, proprietary models for inference, care should be taken to prevent an adversary from stealing the model. The research paper [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/pdf/1609.02943.pdf) outlines experiments performed to show how an attacker can use a prediction API to leak the ML model or construct a nearly identical replica. A simple way to thwart such an attack is to not expose the prediction probabilities to a high degree of precision in the API response.
