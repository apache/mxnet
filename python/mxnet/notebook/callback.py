import datetime
import bokeh.plotting
from collections import defaultdict
import pandas as pd
import time


class PandasLogger(object):
    """Logs statistics about training run into Pandas dataframes.
    Records three separate dataframes: train, eval, epoch.

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        How many training mini-batches between calculations.
        Defaults to calculating every 50 batches.
        (Eval data is stored once per epoch over the entire 
        eval data set.)
    """
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self._dataframes = {
            'train': pd.DataFrame(),
            'eval': pd.DataFrame(),
            'epoch': pd.DataFrame(),
        }
        self.last_time = time.time()
        self.start_time = datetime.datetime.now()
        self.last_epoch_time= datetime.datetime.now()

    @property
    def train_df(self):
        """The dataframe with training data.
        This has metrics for training minibatches, logged every 
        "frequent" batches.  (frequent is a constructor param)
        """
        return self._dataframes['train']

    @property
    def eval_df(self):
        """The dataframe with evaluation data.
        This has validation scores calculated at the end of each epoch.
        """
        return self._dataframes['train']

    @property
    def epoch_df(self):
        """The dataframe with epoch data.
        This has timing information.
        """
        return self._dataframes['train']

    @property
    def all_dataframes(self):
        """Return a dict of dataframes
        """
        return self._dataframes

    def elapsed(self):
        return datetime.datetime.now() - self.start_time

    def _add_new_columns(self, df, metrics):
        #TODO(leodirac): we don't really need to do this on every update.  Optimize
        new_columns = set(metrics.keys()) - set(df.columns)
        for col in new_columns:
            df[col] = None

    def append_metrics(self, metrics, df_name):
        df = self._dataframes[df_name]
        self._add_new_columns(df,metrics)
        df.loc[len(df)] = metrics

    def train_cb(self, param):
        if param.nbatch % self.frequent == 0:
            self._process_batch(param, 'train')

    def eval_cb(self, param):
        self._process_batch(param, 'eval')

    def _process_batch(self, param, df):
        now = time.time()
        if param.eval_metric is not None:
            metrics = dict(param.eval_metric.get_name_value())
            param.eval_metric.reset()
        else:
            metrics = {}
        speed = self.frequent / (now - self.last_time)
        metrics['batches_per_sec'] = speed * self.batch_size
        metrics['records_per_sec'] = speed 
        metrics['elapsed'] = self.elapsed()
        metrics['minibatch_count'] = param.nbatch
        metrics['epoch'] = param.epoch
        self.append_metrics(metrics, df)
        self.last_time = now

    def epoch_cb(self, epoch, symbol, arg_params, aux_params):
        metrics = {}
        metrics['elapsed'] = self.elapsed()
        now = datetime.datetime.now()
        metrics['epoch_time'] = now - self.last_epoch_time
        self.append_metrics(metrics, 'epoch')
        self.last_epoch_time= now

    def callback_args(self):
        """returns **kwargs parameters for model.fit()
        to enable all callbacks.  e.g.
        model.fit(X=train, eval_data=test, **pdlogger.callback_args())
        """
        return {
            'batch_end_callback': self.train_cb,
            'eval_end_callback': self.eval_cb,
            'epoch_end_callback': self.epoch_cb,
        }


class LiveBokehChart(object):
    """Callback object that renders a bokeh chart in a jupyter notebook
    that gets updated as the training run proceeds.

    Requires a PandasLogger to collect the data it will render.

    This is an abstract base-class.  Sub-classes define the specific chart.
    """

    def __init__(self, pandas_logger, metric_name, display_freq=10, 
                 batch_size=None, frequent=50):
        if pandas_logger:
            self.pandas_logger = pandas_logger
        else:
            self.pandas_logger = PandasLogger(batch_size=batch_size, frequent=frequent)
        self.display_freq = display_freq
        self.last_update = time.time() 
        #NOTE: would be nice to auto-detect the metric_name if there's only one.
        self.metric_name = metric_name
        bokeh.io.output_notebook()
        self.handle = self.setup_chart()

    def setup_chart(self):
        """Render a bokeh object and return a handle to it.
        """
        raise NotImplementedError("Incomplete base class: LiveBokehChart must be sub-classed")

    def update_chart_data(self):
        """Update the bokeh object with new data.
        """
        raise NotImplementedError("Incomplete base class: LiveBokehChart must be sub-classed")

    def interval_elapsed(self):
        return time.time() - self.last_update > self.display_freq

    def _push_render(self):
        bokeh.io.push_notebook(handle=self.handle)
        self.last_update = time.time() 

    def _do_update(self):
        self.update_chart_data(self.pandas_logger.all_dataframes)
        self._push_render()

    def batch_cb(self, param):
        if self.interval_elapsed():
            self._do_update()
        
    def eval_cb(self, param):
        # After eval results, force an update.
        self._do_update()

    def callback_args(self):
        """returns **kwargs parameters for model.fit()
        to enable all callbacks.  e.g.
        model.fit(X=train, eval_data=test, **pdlogger.callback_args())
        """
        return {
            'batch_end_callback': self.batch_cb,
            'eval_end_callback': self.eval_cb,
        }


class LiveTimeSeries(LiveBokehChart):

    def __init__(self, **fig_params):
        self.fig = bokeh.plotting.Figure(x_axis_type='datetime', 
                x_axis_label='Elapsed time', **fig_params)
        super(LiveTimeSeries,self).__init__(None,None)  # TODO: clean up this class hierarchy

    def setup_chart(self):
        self.start_time = datetime.datetime.now()
        self.x = []
        self.y = []
        self.fig.line(self.x,self.y)
        return bokeh.plotting.show(self.fig, notebook_handle=True)

    def elapsed(self):
        return datetime.datetime.now() - self.start_time

    def update_chart_data(self, value):
        self.x.append(self.elapsed())
        self.y.append(value)
        self._push_render()


class LiveLearningCurve(LiveBokehChart):
    """Draws a learning curve with training & validation metrics
    over time as the network trains.
    """

    def __init__(self, metric_name, display_freq=10, frequent=50):
        self.display_freq = display_freq
        self.frequent = frequent
        self.last_update = time.time()
        self.start_time = datetime.datetime.now()
        #NOTE: would be nice to auto-detect the metric_name if there's only one.
        self.metric_name = metric_name
        self._data = {
            'train': {'elapsed': [],},
            'eval': {'elapsed': [],},
        }
        bokeh.io.output_notebook()
        self.handle = self.setup_chart()

    def setup_chart(self):
        self.fig = bokeh.plotting.Figure(x_axis_type='datetime', 
                x_axis_label='Training time')
        #TODO(leodirac): There's got to be a better way to 
        # get a bokeh plot to dynamically update as a pandas dataframe changes,
        # instead of copying into a list.
        # I can't figure it out though.  Ask a pyData expert.
        self.x1 = []
        self.y1 = []
        self.train1 = self.fig.line(self.x1,self.y1, line_dash='dotted', 
                alpha=0.3, legend="train")
        self.train2 = self.fig.circle(self.x1,self.y1, size=1.5, 
                line_alpha=0.3, fill_alpha=0.3, legend="train")
        self.train2.visible = False  # Turn this on later.
        self.x2 = []
        self.y2 = []
        self.valid1 = self.fig.line(self.x2,self.y2, line_color='green', line_width=2, legend="validation")
        self.valid2 = self.fig.circle(self.x2,self.y2, line_color='green', line_width=2, legend=None)
        self.fig.legend.location = "bottom_right"
        self.fig.yaxis.axis_label = self.metric_name
        return bokeh.plotting.show(self.fig, notebook_handle=True)

    def _extend(self, a, b):
        """Assuming a is shorter than b, copy the end of b onto a
        """
        a.extend(b[len(a):])

    def _do_update(self):
        self.update_chart_data()
        self._push_render()

    def batch_cb(self, param):
        if param.nbatch % self.frequent == 0:
            self._process_batch(param, 'train')
        if self.interval_elapsed():
            self._do_update()
        
    def eval_cb(self, param):
        # After eval results, force an update.
        self._process_batch(param, 'eval')
        self._do_update()

    def _process_batch(self, param, df_name):
        if param.eval_metric is not None:
            metrics = dict(param.eval_metric.get_name_value())
            param.eval_metric.reset()
        else:
            metrics = {}
        metrics['elapsed'] = datetime.datetime.now() - self.start_time
        for key, value in metrics.items():
            if not self._data[df_name].has_key(key):
                self._data[df_name][key] = []
            self._data[df_name][key].append(value)

    def update_chart_data(self):
        df = self._data['train']
        if len(df['elapsed']):
            self._extend(self.x1, df['elapsed'])
            self._extend(self.y1, df[self.metric_name])
        df = self._data['eval']
        if len(df['elapsed']):
            self._extend(self.x2, df['elapsed'])
            self._extend(self.y2, df[self.metric_name])
        if len(df) > 10:
            self.train1.visible = False
            self.train2.visible = True


def args_wrapper(*args):
    """Generates callback arguments for model.fit()
    for a set of callback objects.
    Callback objects like PandasLogger(), LiveLearningCurve()
    get passed in.  This assembles all their callback arguments.
    """
    pass
    out = defaultdict(list)
    for cb in args:
        callback_args = cb.callback_args()
        for k,v in callback_args.iteritems():
            out[k].append(v)
    return dict(out)


