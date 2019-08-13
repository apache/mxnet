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
""" Helper classes for multiprocess captcha image generation
This module also provides script for saving captcha images to file using CLI.
"""

from __future__ import print_function
import random

import numpy as np
from captcha.image import ImageCaptcha
import cv2
from multiproc_data import MPData


class CaptchaGen(object):
    """Generates a captcha image
    """
    def __init__(self, h, w, font_paths):
        """
        Parameters
        ----------
        h: int
            Height of the generated images
        w: int
            Width of the generated images
        font_paths: list of str
            List of all fonts in ttf format
        """
        self.captcha = ImageCaptcha(fonts=font_paths)
        self.h = h
        self.w = w

    def image(self, captcha_str):
        """Generate a greyscale captcha image representing number string

        Parameters
        ----------
        captcha_str: str
            string a characters for captcha image

        Returns
        -------
        numpy.ndarray
            Generated greyscale image in np.ndarray float type with values normalized to [0, 1]
        """
        img = self.captcha.generate(captcha_str)
        img = np.fromstring(img.getvalue(), dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.h, self.w))
        img = img.transpose(1, 0)
        img = np.multiply(img, 1 / 255.0)
        return img


class DigitCaptcha(object):
    """Provides shape() and get() interface for digit-captcha image generation
    """
    def __init__(self, font_paths, h, w, num_digit_min, num_digit_max):
        """
        Parameters
        ----------
        font_paths: list of str
            List of path to ttf font files
        h: int
            height of the generated image
        w: int
            width of the generated image
        num_digit_min: int
            minimum number of digits in generated captcha image
        num_digit_max: int
            maximum number of digits in generated captcha image
        """
        self.num_digit_min = num_digit_min
        self.num_digit_max = num_digit_max
        self.captcha = CaptchaGen(h=h, w=w, font_paths=font_paths)

    @property
    def shape(self):
        """Returns shape of the image data generated

        Returns
        -------
        tuple(int, int)
        """
        return self.captcha.h, self.captcha.w

    def get(self):
        """Get an image from the queue

        Returns
        -------
        np.ndarray
            A captcha image, normalized to [0, 1]
        """
        return self._gen_sample()

    @staticmethod
    def get_rand(num_digit_min, num_digit_max):
        """Generates a character string of digits. Number of digits are
        between self.num_digit_min and self.num_digit_max
        Returns
        -------
        str
        """
        buf = ""
        max_len = random.randint(num_digit_min, num_digit_max)
        for i in range(max_len):
            buf += str(random.randint(0, 9))
        return buf

    def _gen_sample(self):
        """Generate a random captcha image sample
        Returns
        -------
        (numpy.ndarray, str)
            Tuple of image (numpy ndarray) and character string of digits used to generate the image
        """
        num_str = self.get_rand(self.num_digit_min, self.num_digit_max)
        return self.captcha.image(num_str), num_str


class MPDigitCaptcha(DigitCaptcha):
    """Handles multi-process captcha image generation
    """
    def __init__(self, font_paths, h, w, num_digit_min, num_digit_max, num_processes, max_queue_size):
        """Parameters
        ----------
        font_paths: list of str
            List of path to ttf font files
        h: int
            height of the generated image
        w: int
            width of the generated image
        num_digit_min: int
            minimum number of digits in generated captcha image
        num_digit_max: int
            maximum number of digits in generated captcha image
        num_processes: int
            Number of processes to spawn
        max_queue_size: int
            Maximum images in queue before processes wait
        """
        super(MPDigitCaptcha, self).__init__(font_paths, h, w, num_digit_min, num_digit_max)
        self.mp_data = MPData(num_processes, max_queue_size, self._gen_sample)

    def start(self):
        """Starts the processes"""
        self.mp_data.start()

    def get(self):
        """Get an image from the queue

        Returns
        -------
        np.ndarray
            A captcha image, normalized to [0, 1]
        """
        return self.mp_data.get()

    def reset(self):
        """Resets the generator by stopping all processes"""
        self.mp_data.reset()


if __name__ == '__main__':
    import argparse

    def main():
        """Program entry point"""
        parser = argparse.ArgumentParser()
        parser.add_argument("font_path", help="Path to ttf font file")
        parser.add_argument("output", help="Output filename including extension (e.g. 'sample.jpg')")
        parser.add_argument("--num", help="Up to 4 digit number [Default: random]")
        args = parser.parse_args()

        captcha = ImageCaptcha(fonts=[args.font_path])
        captcha_str = args.num if args.num else DigitCaptcha.get_rand(3, 4)
        img = captcha.generate(captcha_str)
        img = np.fromstring(img.getvalue(), dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(args.output, img)
        print("Captcha image with digits {} written to {}".format([int(c) for c in captcha_str], args.output))

    main()
