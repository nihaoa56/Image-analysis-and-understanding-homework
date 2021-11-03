import torch
import models
import os
import numpy as np
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io

resume = 'ckpt/lr-0.01-iter-490000.pth'
folder = 'result/val/'
all_folder = os.path.join(folder, 'all/')
print(all_folder)
png_folder = os.path.join(folder, 'png/')
mat_folder = os.path.join(folder, 'mat/')
batch_size = 1
assert batch_size == 1


# os.mkdir(all_folder)
# os.mkdir(png_folder)
# os.mkdir(mat_folder)
# except Exception:
#     print('dir already exist....')
#     pass

model = models.resnet101(pretrained=False).cuda()
model.eval()

#resume..
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

test_dataset = BSDS_RCFLoader(split="test")
# print(len(test_dataset))
test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    num_workers=0, drop_last=True, shuffle=False)

if __name__ == "__main__":
    with torch.no_grad():
        for i, (image, ori, img_files) in enumerate(test_loader):
            h, w = ori.size()[2:]
            print(h,w)
            # print()
            image = image.cuda()
            print(image)
            name = img_files[0][5:-4]

            outs = model(image, (h, w))
            fuse = outs[-1].squeeze().detach().cpu().numpy()

            outs.append(ori)

            idx = 0
            print('working on .. {}'.format(i))

            for result in outs:
                idx += 1
                result = result.squeeze().detach().cpu().numpy()
                if len(result.shape) == 3:
                    result = result.transpose(1, 2, 0).astype(np.uint8)
                    result = result[:, :, [2, 1, 0]]
                    Image.fromarray(result).save(os.path.join(all_folder, '{}-img.jpg'.format(name)))
                else:
                    result = (result * 255).astype(np.uint8)
                    # result\val\all
                    Image.fromarray(result).save(all_folder+'{}.png'.format(idx))
            Image.fromarray((fuse * 255).astype(np.uint8)).save(os.path.join(png_folder, '{}.png'.format(name)))
            io.savemat(os.path.join(mat_folder, '{}.mat'.format(name)), {'result': fuse})
        print('finished.')

