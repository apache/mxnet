#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import json
import logging
import subprocess

HOME = os.environ['HOME']
KEY_PATH = os.path.join(HOME, ".m2")


'''
This file would do the following items:
    Import keys from AWS Credential services
    Create settings.xml in .m2 with pass phrase
    Create security-settings.xml in .m2 with master password
    Import keys.asc the encrypted keys in gpg
'''


def getCredentials():
    import boto3
    import botocore
    secret_name = os.environ['DOCKERHUB_SECRET_NAME']
    endpoint_url = os.environ['DOCKERHUB_SECRET_ENDPOINT_URL']
    region_name = os.environ['DOCKERHUB_SECRET_ENDPOINT_REGION']

    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
        endpoint_url=endpoint_url
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except botocore.exceptions.ClientError as client_error:
        if client_error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.exception("The requested secret %s was not found", secret_name)
        elif client_error.response['Error']['Code'] == 'InvalidRequestException':
            logging.exception("The request was invalid due to:")
        elif client_error.response['Error']['Code'] == 'InvalidParameterException':
            logging.exception("The request had invalid params:")
        else:
            raise
    else:
        secret = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret)
        return secret_dict


def importASC(path, passPhrase):
    subprocess.run(['gpg', '--batch', '--yes',
                    '--passphrase=\"{}\"'.format(passPhrase),
                    "--import", "{}".format(os.environ['MVN_DEPLOY_GPG_KEY'])])


def encryptMasterPSW(password):
    result = subprocess.run(['mvn', '--encrypt-master-password', password],
                            stdout=subprocess.PIPE)
    return str(result.stdout)[2:-3]


def encryptPSW(password):
    result = subprocess.run(['mvn', '--encrypt-password', password],
                            stdout=subprocess.PIPE)
    return str(result.stdout)[2:-3]


def masterPSW(password):
    with open(os.path.join(KEY_PATH, "settings-security.xml"), "w") as f:
        f.write("<settingsSecurity>\n <master>{}</master>\n</settingsSecurity>"
                .format(password))


def severPSW(username, password, passPhrase):
    with open(os.path.join(KEY_PATH, "settings.xml"), "w") as f:
        settingsString = '''<?xml version="1.0" encoding="UTF-8"?>
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd">
<pluginGroups></pluginGroups>
<proxies></proxies>
<servers>
<server>
        <id>apache.snapshots.https</id>
        <username>{}</username>
        <password>{}</password>
</server>
<!-- To stage a release of some part of Maven -->
<server>
        <id>apache.releases.https</id>
        <username>{}</username>
        <password>{}</password>
</server>
</servers>
<mirrors></mirrors>
<profiles>
<profile>
        <id>gpg</id>
        <properties>
        <gpg.executable>gpg</gpg.executable>
        <gpg.passphrase>{}</gpg.passphrase>
        </properties>
</profile>
</profiles>
<activeProfiles>
        <activeProfile>gpg</activeProfile>
</activeProfiles>
</settings> '''.format(username, password, username, password, passPhrase)
        f.write(settingsString)


if __name__ == "__main__":
    if not os.path.exists(KEY_PATH):
        os.makedirs(KEY_PATH)
    userCredential = {
        "username": os.environ['MVN_DEPLOY_USER'],
        "password": os.environ['MVN_DEPLOY_PASSWORD']
    }
    keyCredential = {
        "passPhrase": os.environ['MVN_DEPLOY_GPG_PASSPHRASE'],
        "masterPass": os.environ['MVN_DEPLOY_MASTERPASS']
    }
    masterPass = encryptMasterPSW(keyCredential["masterPass"])
    masterPSW(masterPass)
    passwordEncrypted = encryptPSW(userCredential["password"])
    severPSW(userCredential["username"], passwordEncrypted,
             keyCredential["passPhrase"])
    importASC(HOME, keyCredential["passPhrase"])
