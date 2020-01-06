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
Module: preprocess_data
Reference: https://github.com/rizkiarm/LipNet
"""

# pylint: disable=too-many-locals, no-self-use, c-extension-no-member

import os
import fnmatch
import errno
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from skimage import io
import skvideo.io
import dlib

def mkdir_p(path):
    """
    Make a directory
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def find_files(directory, pattern):
    """
    Find files
    """
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

class Video(object):
    """
    Preprocess for Video
    """
    def __init__(self, vtype='mouth', face_predictor_path=None):
        if vtype == 'face' and face_predictor_path is None:
            raise AttributeError('Face video need to be accompanied with face predictor')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype
        self.face = None
        self.mouth = None
        self.data = None
        self.length = None

    def from_frames(self, path):
        """
        Read from frames
        """
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        self.handle_type(frames)
        return self

    def from_video(self, path):
        """
        Read from videos
        """
        frames = self.get_video_frames(path)
        self.handle_type(frames)
        return self

    def from_array(self, frames):
        """
        Read from array
        """
        self.handle_type(frames)
        return self

    def handle_type(self, frames):
        """
        Config video types
        """
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Video type not found')

    def process_frames_face(self, frames):
        """
        Preprocess from frames using face detector
        """
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)
        if mouth_frames[0] is not None:
            self.set_data(mouth_frames)

    def process_frames_mouth(self, frames):
        """
        Preprocess from frames using mouth detector
        """
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)

    def get_frames_mouth(self, detector, predictor, frames):
        """
        Get frames using mouth crop
        """
        mouth_width = 100
        mouth_height = 50
        horizontal_pad = 0.19
        normalize_ratio = None
        mouth_frames = []
        for frame in frames:
            dets = detector(frame, 1)
            shape = None
            for det in dets:
                shape = predictor(frame, det)
                i = -1
            if shape is None: # Detector doesn't detect face, just return None
                return [None]
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x, part.y))
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - horizontal_pad)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + horizontal_pad)

                normalize_ratio = mouth_width / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio),
                             int(frame.shape[1] * normalize_ratio))
            resized_img = imresize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - mouth_width / 2)
            mouth_r = int(mouth_centroid_norm[0] + mouth_width / 2)
            mouth_t = int(mouth_centroid_norm[1] - mouth_height / 2)
            mouth_b = int(mouth_centroid_norm[1] + mouth_height / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

            mouth_frames.append(mouth_crop_image)
        return mouth_frames

    def get_video_frames(self, path):
        """
        Get video frames
        """
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        return frames

    def set_data(self, frames):
        """
        Prepare the input of model
        """
        data_frames = []
        for frame in frames:
            #frame H x W x C
            frame = frame.swapaxes(0, 1) # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1) # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames) # T x W x H x C
        data_frames = np.rollaxis(data_frames, 3) # C x T x W x H
        data_frames = data_frames.swapaxes(2, 3) # C x T x H x W  = NCDHW

        self.data = data_frames
        self.length = frames_n

def preprocess(from_idx, to_idx, _params):
    """
    Preprocess: Convert a video into the mouth images
    """
    source_exts = '*.mpg'
    src_path = _params['src_path']
    tgt_path = _params['tgt_path']
    face_predictor_path = './shape_predictor_68_face_landmarks.dat'

    succ = set()
    fail = set()
    for idx in range(from_idx, to_idx):
        s_id = 's' + str(idx) + '/'
        source_path = src_path + '/' + s_id
        target_path = tgt_path + '/' + s_id
        fail_cnt = 0
        for filepath in find_files(source_path, source_exts):
            print("Processing: {}".format(filepath))
            filepath_wo_ext = os.path.splitext(filepath)[0].split('/')[-2:]
            target_dir = os.path.join(tgt_path, '/'.join(filepath_wo_ext))

            if os.path.exists(target_dir):
                continue

            try:
                video = Video(vtype='face', \
                                face_predictor_path=face_predictor_path).from_video(filepath)
                mkdir_p(target_dir)
                i = 0
                if video.mouth[0] is None:
                    continue
                for frame in video.mouth:
                    io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)
                    i += 1
            except ValueError as error:
                print(error)
                fail_cnt += 1
        if fail_cnt == 0:
            succ.add(idx)
        else:
            fail.add(idx)
    return (succ, fail)

if __name__ == '__main__':
    import argparse
    from multi import multi_p_run, put_worker
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--src_path', type=str, default='../data/mp4s')
    PARSER.add_argument('--tgt_path', type=str, default='../data/datasets')
    PARSER.add_argument('--n_process', type=int, default=1)
    CONFIG = PARSER.parse_args()
    N_PROCESS = CONFIG.n_process
    PARAMS = {'src_path':CONFIG.src_path,
              'tgt_path':CONFIG.tgt_path}

    os.makedirs('{tgt_path}'.format(tgt_path=PARAMS['tgt_path']), exist_ok=True)

    if N_PROCESS == 1:
        RES = preprocess(0, 35, PARAMS)
    else:
        RES = multi_p_run(35, put_worker, preprocess, PARAMS, N_PROCESS)
