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

# This is a web server built based on Flask framework and AWS Elastic Beanstalk service 
# It will response to http GET/POST requests
from flask import Flask, jsonify, request, send_file
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from Predictor import Predictor
from Trainer import Trainer
import plot_piechart
import timeit
import atexit
import logging
import os.path

logging.basicConfig(level=logging.INFO)

application = Flask(__name__)

if not os.path.exists('/tmp/Classifier.p'):
    trainer = Trainer()
    trainer.train()
predictor = Predictor()

# GET '/'
@application.route('/')
def index():
    return "Hello!  -Bot"


# GET '/issues/<issue>'
# return predictions of an issue
@application.route('/issues/<issue>')
def get_prediction(issue):
    l = predictor.predict([issue])
    return " ".join(l[0])


# POST '/predict'
# return predictions of issues
@application.route('/predict', methods=['POST'])
def predict():
    # get prediction results of multiple issues
    # data would be a json file {"issues":[1,2,3]}
    data = request.get_json()["issues"]
    #predictions = predict_labels.predict(data)
    predictions = predictor.predict(data)
    response = []
    for i in range(len(data)):
        response.append({"number":data[i], "predictions":predictions[i]})
    return jsonify(response)


# POST '/draw'
# return an image's binary code
@application.route('/draw', methods=['POST'])
def plot():
    # requests.post(url,json={"fracs":[], "labels":[]})
    data = request.get_json()
    fracs = data["fracs"]
    labels = data["labels"]
    filename = plot_piechart.draw_pie(fracs, labels)
    return send_file(filename, mimetype='image/png')


# helper function
def train_models():
    start = timeit.default_timer()
    trainer = Trainer()
    trainer.train()
    stop = timeit.default_timer()
    # reload models
    predictor.reload()
    time = int(stop - start)
    logging.info("Training completed! Time cost: {} min, {} seconds".format(str(int(time/60)), str(time%60)))
    return 


# Once the server is running, it will retrain ML models every 24 hours
@application.before_first_request
def initialize():
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(
        func=train_models,
        trigger=IntervalTrigger(hours=24),
        id='Training_Job',
        name='Update models every 24 hours',
        replace_existing=True)
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())


initialize()


# run the app.
if __name__ == "__main__":
    # Set debug to True enables debug output.
    # This 'application.debug = True' should be removed before deploying a production app.
    application.debug = True
    application.threaded = True
    application.run('0.0.0.0', 8000)
