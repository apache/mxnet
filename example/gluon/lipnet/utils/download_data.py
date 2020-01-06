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

"""
Module: download_data
This module provides utilities for downloading the datasets for training LipNet
"""

import os
from os.path import exists
from multi import multi_p_run, put_worker


def download_mp4(from_idx, to_idx, _params):
    """
    download mp4s
    """
    succ = set()
    fail = set()
    for idx in range(from_idx, to_idx):
        name = 's' + str(idx)
        save_folder = '{src_path}/{nm}'.format(src_path=_params['src_path'], nm=name)
        if idx == 0 or os.path.isdir(save_folder):
            continue
        script = "http://spandh.dcs.shef.ac.uk/gridcorpus/{nm}/video/{nm}.mpg_vcd.zip".format( \
                    nm=name)
        down_sc = 'cd {src_path} && curl {script} --output {nm}.mpg_vcd.zip && \
                    unzip {nm}.mpg_vcd.zip'.format(script=script,
                                                   nm=name,
                                                   src_path=_params['src_path'])
        try:
            print(down_sc)
            os.system(down_sc)
            succ.add(idx)
        except OSError as error:
            print(error)
            fail.add(idx)
    return (succ, fail)


def download_align(from_idx, to_idx, _params):
    """
    download aligns
    """
    succ = set()
    fail = set()
    for idx in range(from_idx, to_idx):
        name = 's' + str(idx)
        if idx == 0:
            continue
        script = "http://spandh.dcs.shef.ac.uk/gridcorpus/{nm}/align/{nm}.tar".format(nm=name)
        down_sc = 'cd {align_path} && wget {script} && \
                    tar -xvf {nm}.tar'.format(script=script,
                                              nm=name,
                                              align_path=_params['align_path'])
        try:
            print(down_sc)
            os.system(down_sc)
            succ.add(idx)
        except OSError as error:
            print(error)
            fail.add(idx)
    return (succ, fail)


if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--src_path', type=str, default='../data/mp4s')
    PARSER.add_argument('--align_path', type=str, default='../data')
    PARSER.add_argument('--n_process', type=int, default=1)
    CONFIG = PARSER.parse_args()
    PARAMS = {'src_path': CONFIG.src_path, 'align_path': CONFIG.align_path}
    N_PROCESS = CONFIG.n_process

    if exists('./shape_predictor_68_face_landmarks.dat') is False:
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
                  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2')

    os.makedirs('{src_path}'.format(src_path=PARAMS['src_path']), exist_ok=True)
    os.makedirs('{align_path}'.format(align_path=PARAMS['align_path']), exist_ok=True)

    if N_PROCESS == 1:
        RES = download_mp4(0, 35, PARAMS)
        RES = download_align(0, 35, PARAMS)
    else:
        # download movie files
        RES = multi_p_run(tot_num=35, _func=put_worker, worker=download_mp4, \
                          params=PARAMS, n_process=N_PROCESS)

        # download align files
        RES = multi_p_run(tot_num=35, _func=put_worker, worker=download_align, \
                          params=PARAMS, n_process=N_PROCESS)

    os.system('rm -f {src_path}/*.zip && rm -f {src_path}/*/Thumbs.db'.format( \
              src_path=PARAMS['src_path']))
    os.system('rm -f {align_path}/*.tar && rm -f {align_path}/Thumbs.db'.format( \
              align_path=PARAMS['align_path']))
