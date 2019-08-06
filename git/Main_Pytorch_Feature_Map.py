from torch.nn import functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
import Main

def load_net():
    net = Main.CNN()
    net.load_state_dict(torch.load('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Code\\Classification\\Model\\CNN_aug2'))
    return net

test_data = torch.load('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Code\\Classification\\Preprocess\\test_data_aug.pt')
classes = torch.load('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Code\\Classification\\Preprocess\\flower_classes_aug.pt')

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, num_workers=0)

net = load_net().cuda()
net.eval()
finalconv_name = 'layer'


# hook
feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())


net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
# get weight only from the last layer(linear)
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []

    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam

test = iter(test_loader)
image_tensor, _ = next(test)

image_PIL = transforms.ToPILImage()(image_tensor[0])
image_PIL.save('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Feature\\Result.jpg')

image_tensor = image_tensor.cuda()
logit, feat  = net(image_tensor)
# h_x = logit.argmax(dim=1,keepdim=True).data.squeeze()
h_x = F.softmax(logit, dim=1).data.squeeze()

probs, idx = h_x.sort(0, True)
print(idx[0].item(), classes[idx[0]], probs[0].item())
CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
img = cv2.imread('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Feature\\Result.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Feature\\Result_cam.jpg', result)

