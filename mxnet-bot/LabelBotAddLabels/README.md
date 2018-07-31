# label bot
This bot serves to help non-committers add labels to GitHub issues.

"Hi @mxnet-label-bot, please add labels: [operator, feature request]"

## Setup


#### 1. Store a secret
*Manually Store GitHub credentials as a secret in Secrets Manager. Write down secret name and secret ARN for future use*
* Go wo [AWS Secrets Manager Console](https://console.aws.amazon.com/secretsmanager), click **Store a new secret**
* Select secret type
    1. For **Select secret type**, select **Other types of secrets**
    2. For **Specific the key/value pairs**, store your GitHub ID as **GITHUB_USER** and your GitHub OAUTH token as **GITHUB_OAUTH_TOKEN**
    3. Click **Next**
    <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-07-31+at+12.23.28+PM.png" width="500" height="450"><br>
    </div>
* Name and description
    1. Fill in secret name and description. Write down secret name, it will be used in lambda.
    2. Click **Next**
    <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-07-31+at+12.34.48+PM.png" width="500" height="300"><br>
    </div>
* Configure rotation
    1. Select **Disable automatic rotation**
    2. Click **Next**
* Review
    1. Click **Store**
    2. Click the secret, then we can see secret details. Write down **secret name** and **secret ARN** for serverless configuration.
    <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-07-31+at+1.25.26+PM.png" width="400" height="300"><br>
    </div>

#### 2. Deploy Lambda Function
*Deploy this label bot using the serverless framework*
* Configure ***severless.yml***
    1. Under ***iamRolesStatements***, replace ***Resource*** with the secret ARN 
    2. Under ***environment***
        1. Set ***region_name*** as the same region of your secret.
        2. Replace ***secret_name*** with the secret name.
        3. Replace ***REPO*** with the repo's name you want to test.
* Deploy    
Open terminal, go to current directory. run 
```
serverless deploy
```
Then it will set up those AWS services:
1.	A IAM role for label bot with policies:
```
1.secretsmanager:ListSecrets 
2.secretsmanager:DescribeSecret
3.secretsmanager:GetSecretValue 
4.cloudwatchlogs:CreateLogStream
5.cloudwatchlogs:PutLogEvents
```
One thing to mention: this IAM role only has ***Read*** access to the secret created in step1.
2.	A Lambda function will all code needed.
3.	A CloudWatch event which will trigger the lambda function every 5 minutes.  

#### 3.Play with this bot
* Go to the repo, under an **unlabeled** issue, comment "@mxnet-label-bot, please add labels:[bug]". One thing to mention, this bot can only add labels which **exist** in the repo.
* Go to the lambda function's console, click **Test**. 
* Then labels will be added.
    <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-07-31+at+3.10.26+PM.png" width="600" height="150"><br>
    </div>



