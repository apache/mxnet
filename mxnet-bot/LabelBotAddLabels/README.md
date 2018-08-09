# label bot
This bot serves to help non-committers add labels to GitHub issues.

"Hi @mxnet-label-bot, please add labels: [operator, feature request]"

## Setup
**Make sure that all settings is in the same *AWS region***
#### 1. Store a secret
*Create a secret using Terraform*
* Configure ***variables.tf***
    1. In variable "github_credentials", fill in github_user and github_oauth_token. If you don't want to fill in here, then you can leave them blank.
       After you set up a secret, you can go to [AWS Secrets Manager Console](https://console.aws.amazon.com/secretsmanager) console to manually fill in secret values.
    2. In variable "secret_name", fill in the name of your secret. ie:"github/credentials"
* Run `terraform apply`. It will create the secret. Once setup, it will output the secret ARN. Write it down. 
 <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-08-02+at+9.42.56+PM.png" ><br>
 </div>


#### 2. Deploy Lambda Function
*Deploy this label bot using the serverless framework*
* Configure ***severless.yml***
    1. Under ***iamRolesStatements***, replace ***Resource*** with the secret ARN 
    2. Under ***environment***
        1. Set ***region_name*** as the same region of your secret.
        2. Replace ***secret_name*** with the secret name. (Same as the secret name you set in step1)
        3. Replace ***repo*** with the repo's name you want to test.
* Deploy    
Open terminal, go to current directory. run 
```
serverless deploy
```
Then it will set up those AWS services:
* An IAM role for label bot with policies:
```
1.secretsmanager:ListSecrets 
2.secretsmanager:DescribeSecret
3.secretsmanager:GetSecretValue 
4.cloudwatchlogs:CreateLogStream
5.cloudwatchlogs:PutLogEvents
```
One thing to mention: this IAM role only has ***Read*** access to the secret created in step1.
* A Lambda function will all code needed.
* A CloudWatch event which will trigger the lambda function every 5 minutes.  

#### 3.Play with this bot
* Go to the repo, under an **unlabeled** issue, comment "@mxnet-label-bot, please add labels:[bug]". One thing to mention, this bot can only add labels which **exist** in the repo.
* Go to the lambda function's console, click **Test**. 
* Then labels will be added.
    <div align="center">
        <img src="https://s3-us-west-2.amazonaws.com/email-boy-images/Screen+Shot+2018-07-31+at+3.10.26+PM.png" width="600" height="150"><br>
    </div>



