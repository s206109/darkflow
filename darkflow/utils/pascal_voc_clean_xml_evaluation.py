"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob
import re
import math


def _pp(l): # pretty printing
    for i in l: print('{}: {}'.format(i,l[i]))

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()
        # actual parsing
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        jpg = int(re.sub(r'\D', '',jpg))
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)

        all = list()

        for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text
                if name not in pick:
                        continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))

                # get distance z, hachiya

                xmlbox = obj.find('position')
                z = float(xmlbox.find('z').text)


                # get rotation alpha, sasaki
                xmlbox = obj.find('rotation')
                alpha = float(xmlbox.find('object_angle').text)
                #if alpha < 0:alpha += math.pi
                #alpha = abs(math.cos(alpha))

                current = [name,xn,yn,xx,yx,z,alpha]
                all += [current]
        all.insert(0,jpg)
        add = [all]
        dumps += add
        in_file.close()



    os.chdir(cur_dir)
    #dumpsがAnnotationsを抽出している
    return dumps
