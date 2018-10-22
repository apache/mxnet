# Adversarial examples

This demonstrates the concept of "adversarial examples" from [1] showing how to fool a well-trained CNN.
Adversarial examples are samples where the input has been manipulated to confuse a model (i.e. confident in an incorrect prediction) but where the correct answer still appears obvious to a human.
This method for generating adversarial examples uses the gradient of the loss with respect to the input to craft the adversarial examples.

[1] Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." [arXiv preprint arXiv:1412.6572 (2014)](https://arxiv.org/abs/1412.6572)
