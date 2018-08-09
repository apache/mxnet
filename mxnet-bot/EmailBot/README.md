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

## Setup
Setup this email bot using serverless framework / manually.

### Deploy email bot using serverless framework
* Configure ***serverless.yml***
    1. Under ***provider***, replace ***region*** with your aws region
    2. Under ***environment***
        1. replace ***github_user*** with your github id ie:"CathyZhang0822"
        2. replace ***github_oath_token*** with your READ ONLY access token
        3. replece ***repo*** with the repo's name. ie:"apache/incubator-mxnet"
        4. replace ***sender*** with the sender's email
        5. replace ***recipients*** with recipients emails, seperated by comma. ie:"a@email.com, b@email.com"
        6. replace ***aws_region*** with the same aws region in ***provider***
* Deploy
Open terminal, go to current directory. run
```bash
serverless deploy
```
Then it will set up those AWS services:
	* An IAM role for label bot with policies:
```
1.ses:SendEmail
2.ses:SendTemplagedEmail
3.ses:SendRawEmail 
4.cloudwatchlogs:CreateLogStream
5.cloudwatchlogs:PutLogEvents
```
	* A Lambda function will all code needed.
	* A CloudWatch event which will trigger the lambda function everyday at 14:59 and 18:59 UTC. 
* [Verify Email Addresses](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/verify-email-addresses-procedure.html) Go to AWS console -> SES -> Email Addresses to verify email address.
* Test the Lambda Function. On the lambda function's console, click Test.

### Setup email bot manually

* Set an AWS Lambda Function
    * [Create an AWS Lambda Function](https://docs.aws.amazon.com/lambda/latest/dg/get-started-create-function.html) Go to AWS console -> Lambda -> Create function. 
        * Runtime: select Python3.6
        * Role: Create a new IAM role with SES permissions
    * [Upload code](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model-handler-types.html) Save `EmailBot.py` and `lambda_function.py`, package the two files into a .zip file. Then upload the .zip file into the lambda function.
    * Set Environment Variables. Set your own `github_user`, `github_oauth_token`, `repo`, `sender`, `recipients` and `aws_region` as environmental variables.
    * Add a trigger. Select `CloudWatch Events` from the list on the left. Then configure the trigger. ie. create a new rule with schedule expression `cron(30 2 **?*)`. Then this cloudevent will trigger the lambda function everyday at 2:30(UTC)
* [Verify Email Addresses](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/verify-email-addresses-procedure.html) Go to AWS console -> SES -> Email Addresses to verify email address.
* Test the Lambda Function. On the lambda function's console, click Test.
 








