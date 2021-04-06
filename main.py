import os

import cv2
import tensorflow as tf
import tensorflow_hub as hub

INPUT_DIR_PATH = 'input'
OUTPUT_DIR_PATH = 'output'
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2
PERIOD = 25  # a frame goes into the detector once every [period] frames
MIN_SCORE = 0.5  # the minimal score to detect a person

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']


class VideoManager:
    def __init__(self, video_path, ouput_dir):
        """try to read the file to check it's a video and store some data"""
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.ouput_video_path = os.path.join(ouput_dir, self.video_name) + '.avi'

        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            self.height, self.width, self.channels = frame.shape
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_can_be_read = True

        except Exception:
            self.video_can_be_read = False
            print("Unable to read video: ", self.video_name)

    def create_video(self, period=25, min_score=0.5):
        """create an output video with boxes around humans"""
        print("="*50)
        if not self.video_can_be_read:
            print("Unable to read video: ", self.video_name)
            return

        print("processing video:", self.video_name)
        print("Number of frames: ", self.total_frame_count)
        print("dimensions: ", self.width, self.height, self.channels)
        print("frame rate: ", self.fps)

        video = cv2.VideoCapture(self.video_path)
        out = cv2.VideoWriter(self.ouput_video_path, self.fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        boxes = []
        while video.isOpened():

            ret, frame = video.read()
            if not ret:  # no frame anymore, end of the video reached
                break

            if frame_count % period == 0:  # detect boxes once every [period] frames
                boxes = self.detect_box(frame, min_score)

            for ymin, xmin, ymax, xmax in boxes:  # draw a box around every person detected
                min_corner = (int(xmin * self.width), int(ymin * self.height))
                max_corner = (int(xmax * self.width), int(ymax * self.height))
                cv2.rectangle(frame, min_corner, max_corner, LINE_COLOR, LINE_THICKNESS)

            out.write(frame)
            frame_count += 1

            # display how much is done in real time
            print("{0}% done".format(int(frame_count / self.total_frame_count * 100)), end='\r')

        out.release()
        video.release()

    def detect_box(self, img, min_score=0.5):
        """Detect any person in an image with the given minimal score and return the bounding boxes"""
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)

        result = {key: value.numpy() for key, value in result.items()}
        boxes = []
        for entity, box, score in zip(result["detection_class_entities"],
                                      result["detection_boxes"],
                                      result["detection_scores"]):
            if entity in [b'Person', b'Man', b'Woman'] and score >= min_score:
                boxes.append(box)
        return boxes


if __name__ == '__main__':
    for root, _, files in os.walk(INPUT_DIR_PATH):
        for name in files:
            file_path = os.path.join(root, name)
            video_manager = VideoManager(file_path, OUTPUT_DIR_PATH)
            video_manager.create_video(PERIOD, MIN_SCORE)
