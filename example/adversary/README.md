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

# Adversarial examples

This demonstrates the concept of "adversarial examples" from [1] showing how to fool a well-trained CNN.
Adversarial examples are samples where the input has been manipulated to confuse a model (i.e. confident in an incorrect prediction) but where the correct answer still appears obvious to a human.
This method for generating adversarial examples uses the gradient of the loss with respect to the input to craft the adversarial examples.

[1] Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." [arXiv preprint arXiv:1412.6572 (2014)](https://arxiv.org/abs/1412.6572)
