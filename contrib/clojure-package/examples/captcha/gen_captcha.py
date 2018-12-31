#!/usr/bin/env python

from captcha.image import ImageCaptcha
import os
import random

length = 4
width = 160
height = 60
IMAGE_DIR = "images"


def random_text():
    return ''.join(str(random.randint(0, 9))
                   for _ in range(length))


if __name__ == '__main__':
    image = ImageCaptcha(width=width, height=height)
    captcha_text = random_text()
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    image.write(captcha_text, os.path.join(IMAGE_DIR, captcha_text + ".png"))
