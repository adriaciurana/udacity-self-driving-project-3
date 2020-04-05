import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle
from train import *

dataset = {}
with open(TRAIN_DATASET_PATH, 'rb') as f:
    dataset['train'] = pickle.load(f)

with open(VAL_DATASET_PATH, 'rb') as f:
    dataset['valid'] = pickle.load(f)

with open(TEST_DATASET_PATH, 'rb') as f:
    dataset['test'] = pickle.load(f)


"""
	EXPLORE
"""
def explore(split, ax, dataset):
    steering = []
    for data in dataset:
        steering.append(25*data['steering'])
    steering = np.array(steering)

    sns.distplot(steering, ax=ax, bins=np.linspace(-25, 25, 11))
    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel(f"Steering {split} distribution (num samples: {len(steering)})", fontsize=12)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-25, 25)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(1, 3, figure=fig)
gs.update(left=0.05, right=1 - 0.05, top=1 - 0.05, bottom=0.15)


for i, split in enumerate(['train', 'valid', 'test']):
	ax = fig.add_subplot(gs[0, i:i+1])
	explore(split, ax, dataset[split])
plt.show()

"""
	Data aug
"""

def mirror(im, steering):
    return im[:, ::-1], - steering
def data_aug_explore(data):
    im = cv2.imread(os.path.join(FOLDER_DATASET_PATH, data['image_path']))[..., ::-1]
    steering = data['steering']

    # Image crop
    im = im[IM_CROP[0][0]:IM_CROP[0][1], IM_CROP[1][0]:IM_CROP[1][1]]

    # Do aug
    # Non modifies steering value
    im = SEQ.augment_images([im], )[0]

    # Modifies steering value
    if np.random.rand() > 0.5:
        im, steering = mirror(im, steering)

    # Steering normalization
    if 'jitter' in data['params']:
        steering += (np.random.uniform(-data['params']['jitter'], data['params']['jitter']) / 50)
    steering = np.clip(steering, -1, 1)

    #print(steering )
    #steering = np.clip(steering, -90, 90) / 90 # Maximal angle
    steering = (steering + 1) / 2

    return im

fig, ax = plt.subplots(5, 5)
for i in range(5):
	for j in range(5):
		idx = int(np.random.uniform(0, len(dataset['train']) - 1))
		ax[i, j].imshow(data_aug_explore(dataset['train'][idx]))
plt.show()