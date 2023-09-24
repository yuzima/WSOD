from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'checkpoints/wsod/vgg16.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')