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
            # If I'll have to print metrics 'average_loss' iterations from now,
            # append a metric so I can start updating that.
            if((batch_num + self.average_loss) % self.display == 0):
                self.append_one()

            # If I'm less than 'average_loss' iteration away from a display step,
            # update the metrics.
            if((batch_num + self.average_loss) % self.display \< self.average_loss):
                self.update_metrics(module, label)

            # If I'm at a display step, print the metrics.
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
