
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This file is served to fetch secret from Secrets Manager
import boto3
from botocore.exceptions import ClientError
import os
import logging
logging.basicConfig(level=logging.INFO)

def get_secret():
    '''
    This method is to fetch secret values
    Please configure secret_name and region_name as environment variables
    '''
    secret_name = os.environ.get("secret_name")
    region_name = os.environ.get("region_name")
    endpoint_url = "https://secretsmanager.{}.amazonaws.com".format(region_name)
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
        endpoint_url=endpoint_url
    )

    try:
        # Decrypted secret using the associated KMS CMK
        # Depending on whether the secret was a string or binary, one of these fields will be populated
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return secret
        else:
            binary_secret_data = get_secret_value_response['SecretBinary']
            return binary_secret_data
    except ClientError as e:
        logging.exception(e.response['Error']['Code'])

