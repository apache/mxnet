# Elastic Beanstalk Web Server

A web server built on [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/) which can response to GET/POST requests and realize self-maintenance. It mainly has 2 features:
  * Train models: it will retrain Machine Learning models every 24 hours automatically using latest data.
  * Predict labels: once it receives GET/POST requests with issues ID, it will send predictions back.

## Set up
*Make sure you are in current directory.*
* Configure Dockerfile: In `Dockerfile`. Set environment variables (last 3 lines) with real `github_user`, `github_oauth_token` and `repo`.
* Open terminal, run:
```bash
zip eb.zip application.py cron.yaml DataFetcher.py \
Dockerfile Dockerrun.aws.json plot_piechart.py Predictor.py SentenceParser.py Trainer.py \
requirements.txt stopwords.txt
```
It will zip all needed files into `eb.zip`
* Manually create a new Elastic Beanstalk application.
    1. Go to AWS Elastic Beanstalk console, click ***Create New Application***. Fill in *Application Name* and *Description*, click ***Create***.
    2. Under ***Select environment tier***, select ***Web server environment***, click ***Select***.
    3. Under **Base configuration**, select **Preconfigured platform**. In its dropdown, select **Docker**. Then select ***Upload your code***, upload `eb.zip`.
    4. Click ***Configure more options***. Modify Intances, in the dropdown of Instance type, select t2.large. Click ***Create Environment*** (No need to select a security group, EB will create one.)
    5. It will take about 2 minutes to setup the environment. 
    6. Once the environment is setup, it will take 5-10 minutes to generate models. 
    7. Write down URL. (ie: http://labelbot-env.pgc55xzpte.us-east-1.elasticbeanstalk.com)
    