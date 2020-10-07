# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from cv2 import imshow as cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.data.datasets import register_coco_instances
import os

#register_coco_instances("my_dataset_train", {}, "/Users/joostscholten/PycharmProjects/Mask_rcnn/instances_train.json", "/Users/joostscholten/PycharmProjects/Mask_rcnn")  #    first path: path to a single .json file containing all annotations of training images, second path: path to training .jpg images   (In a folder called JPEGImages)
#register_coco_instances("my_dataset_val", {}, "/Users/joostscholten/PycharmProjects/Mask_rcnn/instances_val.json", "/Users/joostscholten/PycharmProjects/Mask_rcnn")      #    first path: path to a single .json file containing all annotations of validation images, second path: path to validation .jpg images   (can be the same folder)
register_coco_instances("my_dataset_train", {}, "/home/rekha/Documents/Rekha/Programming/maskRcnnDetectron2/Rawdata/Mergedannotations(COCOformat)/train.json", "/home/rekha/Documents/Rekha/Programming/maskRcnnDetectron2/Rawdata")
register_coco_instances("my_dataset_val", {}, "/home/rekha/Documents/Rekha/Programming/maskRcnnDetectron2/Rawdata/Mergedannotations(COCOformat)/val.json", "/home/rekha/Documents/Rekha/Programming/maskRcnnDetectron2/Rawdata")
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")     # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "Some_saved_Weights.pth")                                  #training from saved weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0009999.pth")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000                                                                                     #number of iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.INPUT.MIN_SIZE_TRAIN = 100
cfg.INPUT.MAX_SIZE_TRAIN = 100
cfg.INPUT.MIN_SIZE_TEST = 100
cfg.INPUT.MAX_SIZE_TEST = 100

#added by Joost
cfg.TEST.NMS = 0.3                                                                  #default = 0.3
cfg.TEST.DETECTIONS_PER_IMAGE = 100                                                  #default = 100
#cfg.MODEL.DEVICE = 'cpu'                                                            #Line added to run on laptop (wihout GPU)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
#DefaultTrainer.test(cfg=cfg,model=,evaluators=COCOEvaluator)

print(cfg.dump())

from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
#test model on image
chicken_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

from detectron2.utils.visualizer import ColorMode
for d in random.sample(dataset_dicts, 0):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=chicken_metadata, scale=0.4)
    vis = visualizer.draw_dataset_dict(d)
    image = vis.get_image()[:, :, ::-1]
    # cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
    cv2.imshow(' ', image)
    cv2.waitKey(2500)
cv2.destroyAllWindows()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()



cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75                                          #testing confidence threshold
cfg.DATASETS.TEST = ("my_dataset_val", )
predictor = DefaultPredictor(cfg)



#test model on image
chicken_metadata = MetadataCatalog.get("my_dataset_val")
dataset_dicts = DatasetCatalog.get("my_dataset_val")

from detectron2.utils.visualizer import ColorMode
i = 0
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    #im = cv2.imread("Bin_test/8left_Color.jpg")                                        #Line added for bin picking experiment to show result on a specific image
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=chicken_metadata,
                   scale=0.4,
                   instance_mode=ColorMode.IMAGE_BW                                     #remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = v.get_image()[:, :, ::-1]
    # cv2.imshow(d["file_name"], v.get_image()[:, :, ::-1])
    cv2.imshow(' ', image)
    filename="maskOutputImages/image%i.jpg"%i
    cv2.imwrite(filename, image)
    i+=1
    cv2.waitKey(3000)
cv2.destroyAllWindows()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
trainer = DefaultTrainer(cfg) # David: Create a new trainer to prevent memory error
trainer.resume_or_load(resume=True) # David: load weights
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test
# trainer.test(cfg, trainer.model, evaluators=evaluator)

