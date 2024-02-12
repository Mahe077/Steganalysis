"""This module is used to test the XuNet model."""
from glob import glob
import torch
import numpy as np
import imageio as io
from model.model import XuNet

TEST_BATCH_SIZE = 40
COVER_PATH = "data/cover_test/*.png"
STEGO_PATH = "data/stego_test/*.png"
CHKPT = "./checkpoints/net_1.pt"

cover_image_names = glob(COVER_PATH)
stego_image_names = glob(STEGO_PATH)

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = XuNet().cuda()

ckpt = torch.load(CHKPT)
model.load_state_dict(ckpt["model_state_dict"])
# pylint: disable=E1101
images = torch.empty((TEST_BATCH_SIZE, 3, 256, 256), dtype=torch.float)

# pylint: enable=E1101
test_accuracy = []

for idx in range(0, len(cover_image_names), TEST_BATCH_SIZE // 2):
    cover_batch = cover_image_names[idx : idx + TEST_BATCH_SIZE // 2]
    stego_batch = stego_image_names[idx : idx + TEST_BATCH_SIZE // 2]

    batch = []
    batch_labels = []

    xi = 0
    yi = 0
    for i in range(2 * len(cover_batch)):
        if i % 2 == 0:
            batch.append(stego_batch[xi])
            batch_labels.append(1)
            xi += 1
        else:
            batch.append(cover_batch[yi])
            batch_labels.append(0)
            yi += 1
    # pylint: disable=E1101
    for i in range(TEST_BATCH_SIZE):
        image = io.imread(batch[i])
        image = np.transpose(image, (2, 0, 1))  # transpose to [3, 256, 256]
        #images[i, :, :, :] = torch.tensor(image).cuda()
        images[i, :, :, :] = torch.tensor(np.transpose(io.imread(batch[i]), (2, 0, 1))).cuda()

    image_tensor = images.cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
    # pylint: enable=E1101
    outputs = model(image_tensor)
    prediction = outputs.data.max(1)[1]

    accuracy = (
        prediction.eq(batch_labels.data).sum()
        * 100.0
        / (batch_labels.size()[0])
    )
    test_accuracy.append(accuracy.item())

print("test_accuracy = {:.2f}".format(sum(test_accuracy)/len(test_accuracy)))
#print("test_accuracy = {:.2f}".format(36.7))
