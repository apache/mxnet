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
    endpoint_url = os.environ['MAVEN_PUBLISH_SECRET_ENDPOINT_URL']
    secret_creds_name = os.environ['MAVEN_PUBLISH_SECRET_NAME_CREDENTIALS']
    secret_key_name = os.environ['MAVEN_PUBLISH_SECRET_NAME_GPG']
    region_name = os.environ['DOCKERHUB_SECRET_ENDPOINT_REGION']

    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
        endpoint_url=endpoint_url
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_creds_name
        )
        get_secret_key_response = client.get_secret_value(
            SecretId=secret_key_name
        )
    except botocore.exceptions.ClientError as client_error:
        if client_error.response['Error']['Code'] == 'ResourceNotFoundException':
            name = (secret_key_name if get_secret_value_response
                    else secret_creds_name)
            logging.exception("The requested secret %s was not found", name)
        elif client_error.response['Error']['Code'] == 'InvalidRequestException':
            logging.exception("The request was invalid due to:")
        elif client_error.response['Error']['Code'] == 'InvalidParameterException':
            logging.exception("The request had invalid params:")
        raise
    else:
        secret = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret)
        secret_key = get_secret_key_response['SecretString']
        return secret_dict, secret_key


def importASC(key, gpgPassphrase):
    filename = os.path.join(KEY_PATH, "key.asc")
    with open(filename, 'w') as f:
        f.write(key)
    subprocess.check_output(['gpg2', '--batch', '--yes',
                    '--passphrase-fd', '0',
                    "--import", "{}".format(filename)],
                   input=str.encode(gpgPassphrase))


def encryptMasterPSW(password):
    filename = os.path.join(KEY_PATH, "encryptMasterPassword.exp")
    with open(filename, 'w') as f:
        f.write('''
        spawn mvn --encrypt-master-password
        expect -exact "Master password: "
        send -- "{}\r"
        expect eof
        '''.format(password))
    result = subprocess.check_output(['expect', filename])
    return str(result).split('\r\n')[-1][2:-3]


def encryptPSW(password):
    filename = os.path.join(KEY_PATH, "encryptPassword.exp")
    with open(filename, 'w') as f:
        f.write('''
        spawn mvn --encrypt-password
        expect -exact "Password: "
        send -- "{}\r"
        expect eof
        '''.format(password))
    result = subprocess.check_output(['expect', filename])
    return str(result).split('\r\n')[-1][2:-3]


def masterPSW(password):
    with open(os.path.join(KEY_PATH, "settings-security.xml"), "w") as f:
        f.write("<settingsSecurity>\n <master>{}</master>\n</settingsSecurity>"
                .format(password))


def serverPSW(username, password, gpgPassphrase):
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
        <gpg.executable>gpg2</gpg.executable>
        <gpg.passphrase>{}</gpg.passphrase>
        <gpg.skip>true</gpg.skip>
        </properties>
</profile>
</profiles>
<activeProfiles>
        <activeProfile>gpg</activeProfile>
</activeProfiles>
</settings> '''.format(username, password, username, password, gpgPassphrase)
        f.write(settingsString)


if __name__ == "__main__":
    if not os.path.exists(KEY_PATH):
        os.makedirs(KEY_PATH)
    credentials, gpgKey = getCredentials()
    masterPass = encryptMasterPSW(credentials['masterpass'])
    masterPSW(masterPass)
    passwordEncrypted = encryptPSW(credentials['password'])
    serverPSW(credentials['user'], passwordEncrypted,
             credentials['gpgPassphrase'])
    importASC(gpgKey, credentials['gpgPassphrase'])
