# label_bot_predict_labels
This bot will send daily [GitHub issue](https://github.com/apache/incubator-mxnet/issues) reports with predictions of unlabeled issues.
It contains 2 parts:
* Machine Learning part:
  A web server built based on [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/) which can response to GET/POST requests and realize self-maintenance. It mainly has 2 features:
  * Train models: it will retrain Machine Learning models every 24 hours automatically using latest data.
  * Predict labels: once it receives GET/POST requests with issues ID, it will send predictions back.
* Send Daily Emails: 
  An AWS Lambda function which will be triggered everyday. 
  Once this lambda function is executed, it will send POST requests to the Elastic Beanstalk web server asking predictions. 
  Then it will generate email content and send email.

