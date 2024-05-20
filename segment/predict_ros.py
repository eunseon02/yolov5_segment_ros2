# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from message_filters import ApproximateTimeSynchronizer, Subscriber

# image_buffer=[]
MAX_BUFFER_SIZE=5


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    # height = im.height
    # width = im.width
    # shape = (height, width)
    shape = im.shape[:2]  # current shape [height, width]
    # print("shape", shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



class ImageNode(Node):
    def __init__(self):
        super().__init__('NODE')
        print("node")
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.callback,
            10
        )
        self.image_buffer=[]
        # self.time_sub = Subscriber(self, Clock, '/clock')
        # self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.time_sub], queue_size=10, slop=0.1)
        # print("test")


        self.img_size = 640
        self.stride = 32
        self.auto = True


    def callback(self, rgb_msg):
        print("callback")
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9


        rgb_image = self.ros_img_to_numpy(rgb_msg)
        print("rab_image",  rgb_image.shape)


        im, _, _ = letterbox(rgb_image.copy(), self.img_size, stride=self.stride, auto=self.auto)  # resize
        im = im[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
        im = np.expand_dims(im, axis=0) # add batch dim
        im = np.ascontiguousarray(im)  # contiguous
        print("^^",  im.shape)



        if len(self.image_buffer) >= MAX_BUFFER_SIZE:
            self.image_buffer.pop(0)

        print("insert")

        self.image_buffer.append((im, rgb_image, timestamp))

    def ros_img_to_numpy(self, img_msg):

        # Convert ROS Image message to NumPy array
        np_arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        return np_arr



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    # rgb='/camera/aligned_depth_to_color/image_raw',
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-seg",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    print("run")
    rclpy.init(args=None)
    node = ImageNode()

    try:
        while rclpy.ok():
            # rclpy.spin_once(node, timeout_sec=0.1)
            # rclpy.spin(node)
            print("in")
            source = str(source)
            save_img = not nosave and not source.endswith(".txt")  # save inference images
            webcam = source.isnumeric() or source.endswith(".streams")


            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            # print('#########3333', imgsz)

            # Dataloader
            bs = 1  # batch_size
            count = 0
            if webcam:
                view_img = check_imshow(warn=True)
                count += 1
                pass

            else:
                print("error: not connected")

            # Run inference
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

            print("**8", len(node.image_buffer))
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                # print("*****", len(node.image_buffer))
                while node.image_buffer:
                    im, im0s, timestamp = node.image_buffer.pop(0)
                    # im (1, 3, 480, 640)
                    # im0s (480, 640, 3)
                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim


                    # Inference
                    with dt[1]:
                        visualize = False
                        with torch.no_grad():
                            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


                    # Process predictions

                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        s = f'{timestamp} '
                        if webcam:  # batch_size >= 1
                            im0, frame = im0s.copy(), count
                            cv2.imwrite("door_handle.jpg", im0)
                            s += f"{i}: "
                        else:
                            print("error")

                        # p = Path(p)  # to Path
                        # txt_path = str(save_dir / "labels" / p.stem) + (f"_{frame}")  # im.txt
                        s += "%gx%g " % im.shape[2:]  # print string
                        height, width = im.shape[2:]

                        print("w222", im0.shape)
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            if retina_masks:
                                # scale bbox first the crop masks
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                                masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                            else:
                                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                            # Segments
                            if save_txt:
                                segments = [
                                    scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                                    for x in reversed(masks2segments(masks))
                                ]

                            # Print results
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Mask plotting
                            annotator.masks(
                                masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
                                / 255
                                if retina_masks
                                else im[i],
                            )

                            # Write results
                            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    print()
                                    # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

                                    if conf >= 0.6 and names[int(c)]=="door-handle":
                                        xyxy = [int(x.item()) for x in xyxy]
                                        mask = np.zeros((height, width), dtype=bool)
                                        mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 1
                                        mask_uint8 = mask.astype(np.uint8) * 255
                                        cv2.imwrite("bbox_mask.png", mask_uint8)


                                        with open('/home/lee/data.json', 'w') as f:
                                            json.dump(xyxy, f)
                                        # cv2.imwrite("door_handle.jpg", im0)
                                        print("detect!!!!!!!!!11")
                                        torch.cuda.empty_cache()
                                        sys.exit(1)


                        # Stream results
                        im0 = annotator.result()
                        print("result")

                        cv2.imshow(str(1), im0)
                        print("window", im0.shape)


                        if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                            exit()


                    # Print time (inference-only)
                    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    except KeyboardInterrupt:
        pass




    # # Print results
    # print("seen", seen)
    # t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    # LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)


    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line options for YOLOv5 inference including model paths, data sources, inference settings, and
    output preferences.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # parser.add_argument("--rgd", type=str, default='/camera/aligned_depth_to_color/image_raw')
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="whether to plot masks in native resolution")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking for requirements before launching."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

    # rclpy.init(args=[])
    # node = ImageNode()

    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     run(**vars(opt))



if __name__ == "__main__":
    opt = parse_opt()

    main(opt)
