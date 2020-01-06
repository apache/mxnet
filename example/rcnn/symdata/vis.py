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

def vis_detection(im_orig, detections, class_names, thresh=0.7):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    import matplotlib.pyplot as plt
    import random
    plt.imshow(im_orig)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        cls = int(cls)
        if cls > 0 and conf > thresh:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')
    plt.show()
