# Send Daily Reports

Automatically send daily [GitHub issue](https://github.com/apache/incubator-mxnet/issues) reports of repo using [Amazon Simple Email Service](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/quick-start.html) and [AWS Lambda](https://aws.amazon.com/lambda/).

## Set up
*Deploy the lambda function using the serverless framework*
* Configure ***serverless.yml***
    1. Under ***provider***, replace ***region*** with your aws region
    2. Under ***environment***
        1. replace ***github_user*** with your github id ie:"CathyZhang0822"
        2. replace ***github_oath_token*** with your READ ONLY access token
        3. replece ***repo*** with the repo's name. ie:"apache/incubator-mxnet"
        4. replace ***sender*** with the sender's email
        5. replace ***recipients*** with recipients emails, seperated by comma. ie:"a@email.com, b@email.com"
        6. replace ***aws_region*** with the same aws region in ***provider***
        7. replace ***eb_url*** with your Elastic Beanstalk application's URL
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
* A CloudWatch event which will trigger the lambda function everyday at 14:59 UTC. 

##Send Test Email
* Go to this lambda function's console, make sure environment variables are filled in correctly. click **Test**
* Then you will receive the email:
    <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-08-04+at+7.00.52+PM.png"><br>
    </div>

