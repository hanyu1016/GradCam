import argparse
import cv2
import numpy as np
import torch
import os
import re
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    
    parser.add_argument(
        '--image_file_path',
        type=str,
        default= './TestImage', 
        help='Input image file path' )
    parser.add_argument('--source_file_path',type=str,default= '', help='Source image file path' )
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model = models.resnet18()
    source_weight_file_path = args.source_file_path
    # checkpoint = torch.load("C:/Users/MVCLAB/Desktop/DRA/experiment/CA/experiment_candle_CA/CA_candle_model.pkl")
    checkpoint = torch.load(source_weight_file_path)

    model.load_state_dict(checkpoint,False)

    target_layers = [model.layer4]

    file_list =[]
    file_path = args.image_file_path
    
    anomaly_class= file_path.split('/')
    os.mkdir(anomaly_class[7])
    save_target_file_path = 'C:/Users/MVCLAB/Desktop/GradCam/pytorch-grad-cam' +'/'+anomaly_class[7]

    for file in os.listdir(file_path):
        file_list.append(file)

    image_count = len(file_list)
    image_file_count =0
    for image_path_name in file_list:
        image_path_dir = os.path.join(file_path,image_path_name)
        image_file_count += 1

        rgb_img = cv2.imread(image_path_dir, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        targets = None

        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

            cam.batch_size = 8
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        stringName = re.sub(".JPG","",image_path_name)

        cv2.imwrite(os.path.join(save_target_file_path, f'{stringName}_{args.method}_cam.jpg'), cam_image)
        # cv2.imwrite(f'{stringName}_{args.method}_gb.jpg', gb)
        # cv2.imwrite(f'{stringName}_{args.method}_cam_gb.jpg', cam_gb)
