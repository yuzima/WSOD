import json
import shutil

trainToTest = [
    'c6911883-1843-3727-8eaa-41dc8cda8993',
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
    'ff78e1a3-6deb-34a4-9a1f-b85e34980f06',
]

valToTest = [
    '39556000-3955-3955-3955-039557148672',
    'e9a96218-365b-3ecd-a800-ed2c4c306c78',
    'cb0cba51-dfaf-34e9-a0c2-d931404c3dd8',
    '00c561b9-2057-358d-82c6-5b06d76cebcf',
    '64724064-6472-6472-6472-764725145600',
]

testData = {}
toTestImages = []
toTestAnnotations = []

def migrateToTest(toTestSNames, source, sidOffset = 0):
    # Opening val annotation JSON file
    f_ann = open('data/Argoverse/Argoverse-HD/annotations/' + source + '.json')
    data = json.load(f_ann)

    newImages = []
    newAnnotations = []
    toTestSids = [None] * len(toTestSNames)
    toTestSeqDirs = [None] * len(toTestSNames)
    toTestImageIds = []
    toTestImageNewIds = []

    if source == 'val':
        testData['categories'] = data['categories']
        testData['coco_subset'] = data['coco_subset']
        testData['coco_mapping'] = data['coco_mapping']

    # Iterating through the sequences and seq_dirs
    for idx, s in enumerate(data['sequences']):
        if s in toTestSNames:
            pos = toTestSNames.index(s)
            toTestSids[pos] = idx

    for idx, dir in enumerate(data['seq_dirs']):
        if idx in toTestSids:
            pos = toTestSids.index(idx)
            toTestSeqDirs[pos] = dir.replace(source, "test")

    # Filter the images with the sid in moving to test sid list
    for img in data['images']:
        if img['sid'] in toTestSids:
            pos = toTestSids.index(img['sid'])
            img['sid'] = pos + sidOffset
            # Save the old image id
            toTestImageIds.append(img['id'])
            # Generate the new image id in test set
            img['id'] = len(toTestImages) + 1
            toTestImageNewIds.append(img['id'])
            toTestImages.append(img)
        else:
            newImages.append(img)
    data['images'] = newImages

    for ann in data['annotations']:
        if ann['image_id'] in toTestImageIds:
            # Update the image_id to new id in test set
            pos = toTestImageIds.index(ann['image_id'])
            ann['image_id'] = toTestImageNewIds[pos]
            ann['id'] = len(toTestAnnotations) + 1
            toTestAnnotations.append(ann)
        else:
            newAnnotations.append(ann)
    data['annotations'] = newAnnotations

    # Closing val annotation file
    f_ann.close()

    with open('data/Argoverse/Argoverse-HD/annotations/' + source + '_new.json', 'w') as json_file:
        json.dump(data, json_file)

    # Move the image folders from val to test
    for dir in toTestSeqDirs:
        dir = dir.replace("test/", "")
        dir = dir.replace("/ring_front_center", "")
        print('data/Argoverse/Argoverse-1.1/tracking/' + source + '/' + dir + ' -> ' + 'data/Argoverse/Argoverse-1.1/tracking/test/' + dir)
        shutil.move('data/Argoverse/Argoverse-1.1/tracking/' + source + '/' + dir, 'data/Argoverse/Argoverse-1.1/tracking/test/' + dir)

    return toTestSNames, toTestSeqDirs

toTestSNames, toTestSeqDirs = migrateToTest(valToTest, 'val')

testData['sequences'] = toTestSNames
testData['seq_dirs'] = toTestSeqDirs

toTestSNames, toTestSeqDirs = migrateToTest(trainToTest, 'train', len(valToTest))

testData['sequences'].extend(toTestSNames)
testData['seq_dirs'].extend(toTestSeqDirs)

testData['images'] = toTestImages
testData['annotations'] = toTestAnnotations

with open("data/Argoverse/Argoverse-HD/annotations/test.json", "w") as json_file:
    json.dump(testData, json_file)

