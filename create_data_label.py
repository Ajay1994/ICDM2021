import pandas as pd
import os.path as path
from PIL import Image

# labels = pd.read_csv("nih_labels.csv")
# data_file = "nih_data.csv"
# outF = open(data_file, "w")
# outF.write("image,label\n")
# for i in range(0, len(labels)):
#     file = labels.iloc[i, 0]
#     file_path = "/data/NIH_Xray/images/"+file
#     if path.exists(file_path):
#         outF.write(file_path + "," + labels.iloc[i, 1] + "\n")
# outF.close()

data_frame = pd.read_csv("nih_data.csv",  error_bad_lines=False)
print(data_frame[:3])

from tqdm import tqdm
for i in range(100000, len(data_frame)):
    image = data_frame.iloc[i, 0]
    pil_img = Image.open(image).convert("RGB")
    new = image.replace("images", "image")
    if path.exists(new) == False:
        pil_img.save(new)
    if i % 1000 == 0:
        print(i)