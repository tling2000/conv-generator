import torch
import numpy as np

if __name__ == '__main__':
    image = torch.load('/data2/tangling/conv-generator/data/tiny-imagenet/tiny-imagenet-200/sampled2/image_64.pt')
    print(image.shape)
    image_1000 = image.numpy()
    np.save('/data2/tangling/conv-generator/temp/data/image_s2000.npy',image_1000)
    print(image_1000.shape)