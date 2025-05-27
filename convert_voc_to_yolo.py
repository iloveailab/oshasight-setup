# convert_voc_to_yolo.py
'''
Converting from voc to yolo format
'''
# references:
# https://gist.github.com/Amir22010/a99f18ca19112bc7db0872a36a03a1ec
# https://blog.csdn.net/qq_43757976/article/details/131626067

import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

IMG_DIR = 'datasets/mask/images'
ANN_DIR = 'datasets/mask/annotations'
OUT_DIR = 'datasets/mask'

CLASSES = ['with_mask']  # this dataset only has one class

# we split filenames
all_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
trainval, test = train_test_split(all_files, test_size=0.1, random_state=42)
train, val = train_test_split(trainval, test_size=0.1, random_state=42)
splits = {'train': train, 'valid': val, 'test': test}

# we make dirs
for split in splits:
    os.makedirs(os.path.join(OUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels', split), exist_ok=True)

# we convert
def convert(xml_file, out_file, img_w, img_h):
    root = ET.parse(xml_file).getroot()
    yolo_lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        box = obj.find('bndbox')
        x1 = int(box.find('xmin').text)
        y1 = int(box.find('ymin').text)
        x2 = int(box.find('xmax').text)
        y2 = int(box.find('ymax').text)
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(out_file, 'w') as f:
        f.write('\n'.join(yolo_lines))

# we process all files
for split, files in splits.items():
    for img_name in files:
        name = os.path.splitext(img_name)[0]
        src_img = os.path.join(IMG_DIR, img_name)
        src_xml = os.path.join(ANN_DIR, name + '.xml')
        dst_img = os.path.join(OUT_DIR, 'images', split, img_name)
        dst_txt = os.path.join(OUT_DIR, 'labels', split, name + '.txt')

        tree = ET.parse(src_xml)
        root = tree.getroot()
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)

        shutil.copyfile(src_img, dst_img)
        convert(src_xml, dst_txt, w, h)
