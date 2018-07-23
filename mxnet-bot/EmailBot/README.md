# EmailBot

Automatically send daily [GitHub issue](https://github.com/apache/incubator-mxnet/issues) reports using [Amazon Simple Email Service](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/quick-start.html) and [AWS Lambda](https://aws.amazon.com/lambda/).

## Description
### Architecture
An amazon cloudwatch event will trigger lambda function in a certain frequency(ex: 9am every Monday). Once the lambda function is executed, the issue report will be generated and sent to recipients.   
<div align="center">
  <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Email+bot+-+Page+1.jpeg"><br>
</div>

### Email Content

<div align="center">
    <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-07-23+at+10.57.18+AM.png" width="200" height="200"><br>
</div>

## QuickStart
### Requirments:

```
boto3==1.7.29
botocore==1.10.39
```
* Set an AWS Lambda Function
    * [Create an AWS Lambda Function](https://docs.aws.amazon.com/lambda/latest/dg/get-started-create-function.html) Go to AWS console -> Lambda -> Create function. 
        * Runtime: select Python3.6
        * Role: Create a new IAM role with SES permissions
    * [Upload code](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model-handler-types.html) Save `EmailBot.py` and `lambda_function.py`, package the two files into a .zip file. Then upload the .zip file into the lambda function.
    * Set Environment Variables. Set your own `GITHUB_USER`, `GITHUB_OAUTH_TOKEN` and `REPO` as environmental variables.
    * Add a trigger. Select `CloudWatch Events` from the list on the left. Then configure the trigger. ie. create a new rule with schedule expression `cron(30 2 **?*)`. Then this cloudevent will trigger the lambda function everyday at 2:30(UTC)
* [Verify Email Addresses](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/verify-email-addresses-procedure.html) Go to AWS console -> SES -> Email Addresses to verify email address. Then fill in `sender` and `recipients` in `EmailBot.py`.
* Test the Lambda Function. On the lambda function's console, click Test.
 








