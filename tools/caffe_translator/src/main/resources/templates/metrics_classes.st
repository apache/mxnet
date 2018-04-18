<!
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
!>
class TrainMetrics():

    metric_map = {}

    def __init__(self, display=None, average_loss=1):
        self.average_loss = average_loss
        self.display = display


    def process(self, batch_num, module, label):
        if self.display == None:
            return

        if self.average_loss == 1:
            if batch_num % self.display == 0:
                self.update_metrics(module, label, reset=True)
                self.print_metrics(batch_num)
        else:
            # Metrics must be print 'average_loss' iterations from now.
            # Append a metric which will get updated starting now.
            if((batch_num + self.average_loss) % self.display == 0):
                self.append_one()

            # Less that 'average_loss' iteration away from a display step. Update metrics.
            if((batch_num + self.average_loss) % self.display \< self.average_loss):
                self.update_metrics(module, label)

            # At display step. Print metrics.
            if(batch_num % self.display == 0):
                self.print_metrics(batch_num, remove_heads=True)

    def add(self, metric):
        self.metric_map[metric.name] = [metric]

    def append_one(self):
        for key, lst in self.metric_map.iteritems():
            last_element = lst[-1]
            new_element = copy.deepcopy(last_element)
            new_element.reset()
            lst.append(new_element)

    def update_metrics(self, module, label, reset=False):
        for key, lst in self.metric_map.iteritems():
            for metric in lst:
                if reset:
                    metric.reset()
                module.update_metric(metric, label)

    def print_metrics(self, batch_num, remove_heads=False):

        total_loss = 0
        for key, lst in self.metric_map.iteritems():
                total_loss += lst[0].get()[1]

        logger.info("Iteration %d, loss = %f" % (batch_num, total_loss))

        for key, lst in self.metric_map.iteritems():
            if remove_heads:
                metric = lst.pop(0)
            else:
                metric = lst[0]

            logger.info("    %s" % metric)


class TestMetrics():

    metrics = []

    def add(self, metric):
        self.metrics.append(metric)

    def score_and_print(self, module, itr, num_batch):
        for metric in self.metrics:
            metric.reset()
            module.score(itr, metric, num_batch=num_batch)
            logger.info("    %s" % metric)

<if(display)>
display = <display>
<endif>
<if(average_loss)>
average_loss = <average_loss>
<endif>
train_metrics = TrainMetrics(<if(display)>display=display<endif><if(average_loss)>, average_loss=average_loss<endif>)
test_metrics = TestMetrics()
