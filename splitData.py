import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All Data"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    pass  # If the folder doesn't exist, ignore the error

# --------  Directories to Create -----------

os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# --------  Get the Names  -----------


listNames = os.listdir(inputFolderPath)

uniqueNames = [name.split('.')[0] for name in listNames]  # Extracting file names without extensions
uniqueNames = list(set(uniqueNames))

# --------  Shuffle -----------

random.shuffle(uniqueNames)

# --------  Find the number of images for each folder -----------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# --------  Put remaining images in Training -----------

if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining


# --------  Split the list -----------

lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')


# --------  Copy the files  -----------
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        image_path = f'{inputFolderPath}/{fileName}.jpg'
        label_path = f'{inputFolderPath}/{fileName}.txt'

        # Check if the files exist before copying
        if os.path.exists(image_path) and os.path.exists(label_path):
            shutil.copy(image_path, f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
            shutil.copy(label_path, f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")

# -------- Creating Data.yaml file  -----------
# Update the YAML file creation code as per your requirement
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

with open(f"{outputFolderPath}/data.yaml", 'w') as f:
    f.write(dataYaml)

print("Data.yaml file Created...")
