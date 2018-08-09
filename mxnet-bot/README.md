# MXNet Bot
This is a directory contains bots served to improve operational efficiency.

## Email Bot
Automatically send daily [GitHub issue](https://github.com/apache/incubator-mxnet/issues) reports using [Amazon Simple Email Service](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/quick-start.html) and [AWS Lambda](https://aws.amazon.com/lambda/).

## Label Bot
This bot will help automate/simplify issue labeling process, which mainly contains 3 parts:
* Machine Learning part:
  A web server built based on AWS Elastic Beanstalk which can response to GET/POST requests and realize self-maintenance. It mainly has 2 features:
  * Train models: it will retrain Machine Learning models every 24 hours automatically using latest data.
  * Predict labels: once it receives GET/POST requests with issues ID, it will send predictions back.
* Send Daily Emails: Automatically send daily [GitHub issue](https://github.com/apache/incubator-mxnet/issues) reports listing unlabeled issues and recommended labels.
* Add Labels: An API built using API Gateway and Lambda. Once this API is given correct GitHub credentials, issue ID and labels. It will add labels to corresponding issues.


 








