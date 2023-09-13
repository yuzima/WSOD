#creates YOLOv7 labels for Argoverse

import json
from tqdm import tqdm
from pathlib import Path
import shutil
import os


def argoverse2yolo(set):
    labels = {}
    a = json.load(open(set, "rb"))
    for annot in tqdm(a['annotations'], desc=f"Converting {set} to YOLOv7 format..."):
        img_id = annot['image_id']
        img_name = a['images'][img_id]['name']
        img_label_name = f'{img_name[:-3]}txt'

        cls = annot['category_id']  # instance class id
        x_center, y_center, width, height = annot['bbox']
        x_center = (x_center + width / 2) / 1920.0  # offset and scale
        y_center = (y_center + height / 2) / 1200.0  # offset and scale
        width /= 1920.0  # scale
        height /= 1200.0  # scale

        img_dir = set.parents[2] / 'Argoverse-1.1' / 'labels' / a['seq_dirs'][a['images'][annot['image_id']]['sid']]
        if not img_dir.exists():
            img_dir.mkdir(parents=True, exist_ok=True)

        k = str(img_dir / img_label_name)
        if k not in labels:
            labels[k] = []
        labels[k].append(f"{cls} {x_center} {y_center} {width} {height}\n")

    for k in labels:
        with open(k, "w") as f:
            f.writelines(labels[k])


dir = Path(".")  # dataset root dir

# Convert
annotations_dir = 'Argoverse-HD/annotations/'
(dir / 'Argoverse-1.1' / 'tracking').rename(dir / 'Argoverse-1.1' / 'images')  # rename 'tracking' to 'images'

#default Argoverse train/val split. we still need to split these further to create a (labelled) test set.
for d in "train.json", "val.json":
    argoverse2yolo(dir / annotations_dir / d)  # convert Argoverse annotations to YOLO labels

    
#split out-of-box train/val sets into train/test/val
    
train2test = ['c6911883-1843-3727-8eaa-41dc8cda8993',
 'cd38ac0b-c5a6-3743-a148-f4f7b804ed17',
 'd4d9e91f-0f8e-334d-bd0e-0d062467308a',
 'd60558d2-d1aa-34ee-a902-e061e346e02a',
 'dcdcd8b3-0ba1-3218-b2ea-7bb965aad3f0',
 'de777454-df62-3d5a-a1ce-2edb5e5d4922',
 'e17eed4f-3ffd-3532-ab89-41a3f24cf226',
 'e8ce69b2-36ab-38e8-87a4-b9e20fee7fd2',
 'e9bb51af-1112-34c2-be3e-7ebe826649b4',
 'ebe7a98b-d383-343b-96d6-9e681e2c6a36',
 'f0826a9f-f46e-3c27-97af-87a77f7899cd',
 'f3fb839e-0aa2-342b-81c3-312b80be44f9',
 'fa0b626f-03df-35a0-8447-021088814b8b',
 'fb471bd6-7c81-3d93-ad12-ac54a28beb84',
 'ff78e1a3-6deb-34a4-9a1f-b85e34980f06']

val2test = ['39556000-3955-3955-3955-039557148672',
 'e9a96218-365b-3ecd-a800-ed2c4c306c78',
 'cb0cba51-dfaf-34e9-a0c2-d931404c3dd8',
 '00c561b9-2057-358d-82c6-5b06d76cebcf',
 '64724064-6472-6472-6472-764725145600']

os.mkdir("Argoverse-1.1/images/test/")
os.mkdir("Argoverse-1.1/labels/test/")

for scene in train2test:
    shutil.move(f"Argoverse-1.1/images/train/{scene}", "Argoverse-1.1/images/test/")
    shutil.move(f"Argoverse-1.1/labels/train/{scene}", "Argoverse-1.1/labels/test/")

for scene in val2test:
    shutil.move(f"Argoverse-1.1/images/val/{scene}", "Argoverse-1.1/images/test/")
    shutil.move(f"Argoverse-1.1/labels/val/{scene}", "Argoverse-1.1/labels/test/")
