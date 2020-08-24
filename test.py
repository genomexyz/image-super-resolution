#!/usr/bin/python3

from ISR.models import RRDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.models import RDN
import numpy as np
from PIL import Image

#open data
img = Image.open('data/input/sample/meerkat.png')
lr_img = np.array(img)

#rdn = RDN(weights='psnr-small')
rdn = RDN(weights='psnr-large')
sr_img = rdn.predict(lr_img)
im = Image.fromarray(sr_img)
im.save('meerkat-sr-large.png')

print('before ', np.shape(lr_img))
print('after ', np.shape(sr_img))
