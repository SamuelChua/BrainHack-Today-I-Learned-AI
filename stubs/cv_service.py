import sys
weights_dir = r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower\stubs\model\model_final_66.4%_15k_start_fresh_egg.pth'
from typing import List, Any    

from transformers import YolosConfig
from tilsdk.cv.types import *
#import onnxruntime as ort


root_dir_cv = r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower\installs-new-cv'
# sys.path.insert(0, f'{root_dir}/installs-new')
sys.path.insert(1, root_dir_cv)



import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


from detectron2.checkpoint import DetectionCheckpointer

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
#register_coco_instances("til", {}, "C:\Users\intern\til-final-2022.1.1-lower\new_combined.json", "/content/drive/MyDrive/Brainhack/CV_Collated_Datasets/")
#register_coco_instances("tilval", {}, r"C:\Users\intern\til-final-2022.1.1-lower\qualifiers_finals_no_annotations.json", r"C:\Users\intern\til-final-2022.1.1-lower\Images")

# til_metadata = MetadataCatalog.get("til")
# dataset_dicts=DatasetCatalog.get("til")


#tilval_metadata = MetadataCatalog.get("tilval")
#datasetval_dicts=DatasetCatalog.get("tilval")

from detectron2.engine import DefaultTrainer



cfg = get_cfg()

#cfg.DATASETS.TRAIN = ("til",)
#cfg.DATASETS.TEST = ("tilval",)
cfg.TEST.EVAL_PERIOD= 4000
cfg.SOLVER.CHECKPOINT_PERIOD = 1500

cfg.DATALOADER.NUM_WORKERS = 2
 # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
#cfg.OUTPUT_DIR = "/content/drive/MyDrive/Brainhack/Detectron2 Weights"

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg) 
#trainer.resume_or_load(resume=False)


#DetectionCheckpointer(trainer.model).load(r'C:\Users\intern\til-final-2022.1.1-lower\stubs\model\model_final_66.4%_15k_start_fresh_egg.pth')
cfg.MODEL.WEIGHTS = weights_dir 

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_66.4%_15k_start_fresh_egg.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
COUNT = 0


#YOLO
# import argparse
# import time
# from pathlib import Path

# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from numpy import random
# import os

# # from models.experimental import attempt_load
# # from utils.datasets import LoadStreams, LoadImages
# # from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
# #     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# # from utils.plots import plot_one_box
# # from utils.torch_utils import select_device, load_classifier, time_synchronized


# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#         ('rtsp://', 'rtmp://', 'http://'))

#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Initialize
#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size
#     if half:
#         model.half()  # to FP16

#     # Second-stage classifier
#     classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

#     # Set Dataloader
#     vid_path, vid_writer = None, None
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     else:
#         save_img = True
#         dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Inference
#         t1 = time_synchronized()
#         pred = model(img, augment=opt.augment)[0]

#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t2 = time_synchronized()

#         # Apply Classifier
#         if classify:
#             pred = apply_classifier(pred, modelc, img, im0s)

#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if webcam:  # batch_size >= 1
#                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                     if save_img or view_img:  # Add bbox to image
#                         label = f'{names[int(cls)]} {conf:.2f}'
#                         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.3f}s)')

#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video'
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer

#                         fourcc = 'mp4v'  # output video codec
#                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
#                     vid_writer.write(im0)

#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         print(f"Results saved to {save_dir}{s}")

#     print(f'Done. ({time.time() - t0:.3f}s)')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     opt = parser.parse_args()
#     print(opt)
#     check_requirements()

#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
#                 detect()
#                 strip_optimizer(opt.weights)
#         else:
#             detect()



class CVService:
    def __init__(self, model_dir):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''

        # TODO: Participant to complete.
        self.model_dir = model_dir
        pass

    

    def increment(self):
        global COUNT
        COUNT += 1

    def targets_from_image(self, img) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        l = []
        
        # get predictions for each img
        outputs = predictor(img)
        
        # get num of boxes for each img 
        num_boxes = len(outputs["instances"])

        # get all bbox for 1 img + change format from xmin ymin xmax ymax to xmin ymin w h
        bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist()

        # loop thru all box in each img 
        for boxes in range(num_boxes):
            if num_boxes == 0:
                break
        # 1 box = 1 dic
        #     dic = {'id': '',
        #     'cls': '',
        #     "bbox": {
        #     "x": '',
        #     "y": '',
        #     "w": '',
        #     "h": '',   
        #     }
        # }
        
        ## box label 
            opp_cls = outputs['instances'].pred_classes[boxes].item() + 1

            if opp_cls == 1:
                cls = 1
            else: 
                cls = 0


        # 1 box = 1 dic 
            diff = bboxes[boxes][2] - bboxes[boxes][0]
            new = bboxes[boxes][0] + diff/2
            diff1 = bboxes[boxes][3] - bboxes[boxes][1]
            new1 = bboxes[boxes][1] + diff1/2
            bboxes[boxes][2] = bboxes[boxes][2] - bboxes[boxes][0]
            bboxes[boxes][3]= bboxes[boxes][3] - bboxes[boxes][1]
        
        # dic['bbox'].x = bboxes[boxes][0]
        # dic['bbox'].y = bboxes[boxes][1]
        # dic['bbox'].w = bboxes[boxes][2]
        # dic['bbox'].h = bboxes[boxes][3]
        
        # add to list
        

            bbox = BoundingBox(new,new1,bboxes[boxes][2],bboxes[boxes][3])
            print ("yes")
            obj = DetectedObject(COUNT, cls, bbox)
            self.increment()
            l.append(obj)
            print (l)
        return l


                # TODO: Participant to complete.


class MockCVService:
    '''Mock CV Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        # Does nothing.
        self.model_dir = model_dir
        pass

    def targets_from_image(self, img:Any) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        # dummy data
        print ("helloworld")
        bbox = BoundingBox(100,100,300,50)
        obj = DetectedObject("1", "1", bbox)
        return [obj]