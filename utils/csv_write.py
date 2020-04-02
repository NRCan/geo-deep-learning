from pathlib import Path
from natsort import natsorted
import random
import pandas as pd
import fiona
img_path = Path('/home/maju/data/kingston/RGB')
vector_path = Path('/home/maju/data/kingston/gpkg')
test_percentage = 15
images = natsorted([str(x) for x in img_path.glob('*.tif')])
vectors = [str(x) for x in vector_path.glob('*.gpkg')]
attributes = ['properties/classe' for i in range(len(images))]


k = len(images) * test_percentage // 100
if len(vectors) == 1:  # Todo: Build function for multiple packages to images
    vectors = [vectors[0] for i in range(len(images))]
random.seed(20)
test_indices = random.sample(range(len(images)), k)

images_split = ['trn' for i in range(len(images))]

for i in test_indices:
    images_split[i] = 'tst'

d = {1: images, 2: '', 3: vectors, 4: attributes, 5: images_split}
df = pd.DataFrame(data=d)
df.to_csv('./data/trn_tst_kingston.csv', header=False, index=False)



