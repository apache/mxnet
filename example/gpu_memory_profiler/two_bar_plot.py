#!/usr/bin/python3
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
"""two_bar_plot.py

   Visualize the memory profile using a two-bar plot.
   Run this file after setting the 'SETME.py'.
"""
import numpy as np
import matplotlib.pyplot as plt

from SETME import layer_kw_dict, \
        data_struct_kw_dict, \
        memory_profile_path, expected_sum
from profile_analyzer import analyze


def two_bar_plot(sorted_list_1, sorted_list_2, fig_name):
    """
    Visualize the memory profile in a two-bar plot.
    """
    bar_width, annotation_fontsize = 1, 18
    plt.figure(figsize=(10, 6))

    def _plotList(list, x):
        legend_handles = []
        klist = [kv[0] for kv in list]
        vlist = [kv[1][0] if isinstance(kv[1], type([])) else kv[1] for kv in list]
        for i in range(len(list)):
            legend_handles.append(
                plt.bar(x=x, height=vlist[i], bottom=np.sum(vlist[i+1:]),
                        width=bar_width, edgecolor='black', linewidth=3,
                        color=np.array([1, 0, 0]) if i == 0 else \
                              'white' if 'Other'       in klist[i] or \
                                         'Untrackable' in klist[i] else \
                               np.array([(i-1) * 1.0 / (len(list) - 2),
                                         (i-1) * 1.0 / (len(list) - 2),
                                         (i-1) * 1.0 / (len(list) - 2)]),
                        hatch='//' if klist[i] == 'Untrackable' else '',
                        label=klist[i]))
        return legend_handles

    def _annotateList(list, x, annotation_fontsize, initial_side=True):
        vlist = [kv[1][0] if isinstance(kv[1], type([])) else kv[1] for kv in list]
        side, previous_side = initial_side, initial_side
        for i in range(len(list)):
            middle_pos = vlist[i] / 2 + np.sum(vlist[i+1:])
            bar_length = vlist[i]
            if vlist[i] / sum(vlist) < 0.05:
                continue
            if vlist[i] / sum(vlist) < 0.1:
                side = not previous_side

            plt.annotate('%.0f%%' % (vlist[i] * 100.0 / sum(vlist)),
                         xy=(x+0.6 * (1 if side is True else -1), middle_pos),
                         xytext=(x+0.81 * (1 if side is True else -1), middle_pos),
                         fontsize=annotation_fontsize,
                         ha='left' if side is True else 'right',
                         va='center',
                         bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                         arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" %
                                         (0.5 / plt.ylim()[1] * annotation_fontsize * bar_length),
                                         linewidth=2))
            previous_side = side

    stride = 4 * bar_width
    sorted_list_1_handles = _plotList(sorted_list_1, 0)
    sorted_list_2_handles = _plotList(sorted_list_2, stride)
    _annotateList(sorted_list_1, 0, annotation_fontsize, True)
    _annotateList(sorted_list_2, stride, annotation_fontsize, False)

    # x- & y- axis
    plt.xlim([-4*bar_width, stride + 4*bar_width])
    plt.xticks([0, stride], ["Layer", "Data Structure"])
    plt.xlabel(r"GPU Memory Consuimption Profile")
    plt.ylabel(r"Memory Consumption ($\mathtt{GiB}$)")

    # Grid & Legend
    plt.grid(linestyle='-.', linewidth=1, axis='y')
    legend_artist = plt.legend(handles=sorted_list_1_handles,
                               loc="upper left",
                               fontsize=annotation_fontsize)
    plt.legend(handles=sorted_list_2_handles,
               loc="upper right",
               fontsize=annotation_fontsize)
    plt.gca().add_artist(legend_artist)

    # Tighten Layout and Savefig
    plt.tight_layout()
    plt.savefig(fig_name)


if __name__ == '__main__':
    layer_wise_memory_profile, data_struct_wise_memory_profile = \
            analyze(memory_profile_path,
                    layer_kw_dict,
                    data_struct_kw_dict,
                    expected_sum)

    # initialize the RC parameters
    plt.rc('axes', axisbelow=True)
    plt.rc('mathtext', fontset='cm')
    plt.rc('mathtext', rm='Latin Modern Roman')
    plt.rc('font', family='Latin Modern Roman', size=24)

    two_bar_plot(layer_wise_memory_profile, data_struct_wise_memory_profile, "mxnet_gpu_memory_profile")
