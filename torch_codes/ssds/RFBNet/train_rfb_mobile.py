from rfb_trainer import Trainer

from models.RFB_Net_mobile import build_rfb_mobilenet
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
import torch
from torch.utils.data import DataLoader
from models.RFB_Net_mobile import build_rfb_mobilenet

from alfred.dl.torch.common import device


batch_size = 2
pretrained_model = 'weights/mobilenet_feature.pth'
# pretrained_model = None
num_classes = 81
img_dim = 300
rgb_means = (103.94, 116.78, 123.68)
p = 0.2


def train():
    train_sets = [('2017', 'train'), ('2014', 'val')]
    cfg = COCO_mobile_300

    dataset = COCODetection(COCOroot, train_sets, preproc(img_dim, rgb_means, p))

    net = build_rfb_mobilenet('train', img_dim, num_classes)
    net.to(device)

    if pretrained_model:
        base_weights = torch.load(pretrained_model)
        print('Loading base network...')
        net.base.load_state_dict(base_weights)
        print('Base weights load done.')
    net.train()

    train_loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=detection_collate)

    trainer = Trainer(model=net, train_loader=train_loader, val_loader=None, cfg=cfg, save_epochs=20,
                      checkpoint_dir='./weights', resume_from='checkpoint.pth.tar')
    trainer.train()


if __name__ == '__main__':
    train()