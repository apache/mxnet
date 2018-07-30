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

import datetime
import matplotlib
# set 'agg' as matplotlib backend
matplotlib.use('agg', warn=False, force=True)
from matplotlib import pyplot as plt


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct, v=val)

    return my_autopct


def draw_pie(fracs, labels):
    # plot the pie chart of labels, save the pie chart into '/img' folder 
    fig = plt.figure()
    plt.pie(fracs, labels=labels, autopct=make_autopct(fracs), shadow=True)
    plt.title("Top 10 labels for newly opened issues")
    figname = "piechart_{}_{}.png".format(str(datetime.datetime.today().date()),
                                          str(datetime.datetime.today().time()))
    fig.savefig("/tmp/{}".format(figname))
    pic_path = "/tmp/{}".format(figname)
    return pic_path
