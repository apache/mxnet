### End to end one step ahead time series prediction gluon code for the article:

'Conditional Time Series Forecasting with Convolutional Neural Networks'

https://arxiv.org/abs/1703.04691

and

'Lorenz Trajectories Prediction: Travel Through Time'.
https://arxiv.org/abs/1903.07768

Plotting two of the three trajectories (z vs x) gives rise to the Lorenz butterfly.

![Lorenz_butterfly](assets/butterfly.png)

### Training and inference

Model achieves on average the test RMSE reported in https://arxiv.org/abs/1703.04691). For the unconditional prediction
of the x trajectory, an average of 0.003 is achieved, with some variance.

###### default setting
```
python main.py
``` 

### Learning for Conditional model for x series one step ahead prediction:

![losses_cw](assets/train_loss.png)

### Predictions vs ground truth for x trajectory:

![preds_cwn](assets/predsx_cw.png)
