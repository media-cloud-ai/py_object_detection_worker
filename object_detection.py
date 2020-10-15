import cv2
import json
import logging
import numpy as np
import os
import sys


def get_name():
    return "object_detection"


def get_short_description():
    return "Dectect objects using Yolo model"


def get_description():
    return """Retrieves objects on images.
    It will return a frame by frame analysis.
    The process is base on OpenCV with Yolo model.
    """


def get_version():
    return "0.0.1"


def get_parameters():
    return [
        {
            "identifier": "source_path",
            "label": "Source path",
            "kind": ["string"],
            "required": True,
        },
        {
            "identifier": "destination_path",
            "label": "Destination path",
            "kind": ["string"],
            "required": True,
        },
        {
            "identifier": "confidence",
            "label": "Confidence level of matching detected objects",
            "kind": ["integer"],
            "required": False,
        },
        {
            "identifier": "requirements",
            "label": "Requirements",
            "kind": ["requirements"],
            "required": False,
        }
    ]


def init():
    global knowledgeNetwork, layerNames, yoloLabels

    logLevel = os.environ.get('RUST_LOG', 'warning').upper()
    timestamp = "%(asctime)s.%(msecs)03d000000 UTC"
    logFormat = "%s - - {job_queue:s} -  - %(levelname)s - %(message)s".format(
        timestamp,
        job_queue=os.getenv("AMQP_QUEUE", default="unknown_queue")
    )

    logging.basicConfig(stream=sys.stdout,
                        level=logLevel,
                        format=logFormat,
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )

    labelsPath = "/sources/models/coco.names"
    configPath = "/sources/models/yolov3.cfg"
    weightsPath = "/sources/models/yolov3.weights"

    yoloLabels = open(labelsPath).read().strip().split("\n")

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    knowledgeNetwork = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    layerNames = knowledgeNetwork.getLayerNames()
    layerNames = [
        layerNames[i[0] - 1]
        for i in knowledgeNetwork.getUnconnectedOutLayers()
    ]


def init_process(stream_handler, format_context, parameters):
    global confidence
    video_filters = [
        {
            "name": "format",
            "label": "format_filter",
            "parameters": {
               "pix_fmts": "rgb24"
            }
        }
    ]

    video_stream = stream_handler.new_video_stream(0, video_filters)

    confidence = parameters["confidence"] / 100

    # returns a list of description of the streams to be processed
    return [
        video_stream
    ]


def process_frame(job_id, stream_index, frame):
    global confidence, knowledgeNetwork, layerNames, yoloLabels

    img_np = np.fromstring(frame.data[0], dtype=np.uint8)
    img_np = img_np.reshape((3, frame.width, frame.height), order='F')
    img_np = np.swapaxes(img_np, 0, 2)

    blob = cv2.dnn.blobFromImage(
        img_np,
        1 / 255.0, (512, 512),
        swapRB=False,
        crop=False
    )

    knowledgeNetwork.setInput(blob)

    layerOutputs = knowledgeNetwork.forward(layerNames)
    detectedObjects = []

    # loop over each of the layer outputs
    for layerOutput in layerOutputs:
        # loop over each of the detections
        for detection in layerOutput:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            objectConfidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if objectConfidence > confidence:

                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([
                    frame.width,
                    frame.height,
                    frame.width,
                    frame.height
                ])

                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                detectedObjects.append({
                    'label': yoloLabels[classID],
                    'confidence': float(objectConfidence),
                    'boxe': [x, y, int(width), int(height)]
                })

    return detectedObjects


def ending_process():
    '''
    Function called at the end of the media process
    (the "media" feature must be activated).
    '''
    logging.info("Ending Python worker process...")
