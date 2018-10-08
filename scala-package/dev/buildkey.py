#!/usr/bin/env python
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

import subprocess
import time

'''
Thhis file would do the following items:
	Import keys from AWS Credential services
	Create settings.xml in .m2 with pass phrase
	Create security-settings.xml in .m2 with master password
	Import keys.asc the encrypted keys in gpg
'''

def importASC(path, passPhrase):
	subprocess.run(['gpg', '--batch', '--yes',
	 '--passphrase=\"{}\"'.format(passPhrase),
		"--import", "{}/keys.asc".format(path)])

def encryptMasterPSW(password):
	result = subprocess.run(['mvn', '--encrypt-master-password', password],
	 stdout=subprocess.PIPE)
	return str(result.stdout)[2:-3]

def encryptPSW(password):
	result = subprocess.run(['mvn', '--encrypt-password', password],
	 stdout=subprocess.PIPE)
	return str(result.stdout)[2:-3]

def createASC(path, data):
	f = open(path + "keys.asc", "w+")
	f.write(data)
	f.close()

def masterPSW(path, password):
	f = open(path + "settings-security.xml", "w+")
	f.write("<settingsSecurity>\n <master>{}</master>\n</settingsSecurity>"
		.format(password))
	f.close()

def severPSW(path, username, password, passPhrase):
	f = open(path + "settings.xml", "w+")
	f.write(
		'''
	<?xml version="1.0" encoding="UTF-8"?>
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
      			</properties>
    		</profile>
  			</profiles>
  			<activeProfiles>
    			<activeProfile>gpg</activeProfile>
			</activeProfiles>
		</settings>
          '''.format(username, password, username, password, passPhrase))

if __name__ == "__main__":
	# Assume I got the secret key from secret services
	# secret = get_secret()
	userCredential = {"username": "nswamy", "password": "123456"}
	keyCredential = {
	 "passPhrase" : "asdfg", "encrytedKey": "A-Very-long-string",
	 "masterPass": "123456"}
	masterPass = encryptMasterPSW(keyCredential["masterPass"])
	masterPSW("~/.m2/", masterPass)
	passwordEncrypted = encryptPSW(userCredential["password"])
	severPSW("~/.m2/", userCredential["username"], passwordEncrypted,
	keyCredential["passPhrase"])
	# createASC("./", keyCredential["encrytedKey"])
	# importASC(keyCredential["passPhrase"])

