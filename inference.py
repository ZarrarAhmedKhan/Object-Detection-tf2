""" Generalized Inference Script for Tensorflow's Object Detection API 

This script will evaluate the input (image/folder_of_images/local_video/video_url) and run the inference based on it.

Input Options:
--------------
  * `input_directory`     : input - path to {image/folder/video/} or Valid URL
  * `output_directory`    : output - path to save the output
  * `score_thresh`        : threshold below which API should not consider the objects bounding boxes

Output Options:
---------------
    - Image/Directory/Video

Notes:
------
    None

Example Usage:
--------------
python3 inference.py -i path/to/image.png -o path/to/output.png -t 0.5
python3 inference.py -i path/to/images -o path/to/outputs (should have images)
python3 inference.py -i path/to/video.mp4 -o path/to/output.mp4
python3 inference.py -i https://c.veocdn.com/9408f757-91e3-4fbc-a53b-0ab2d3c5e64f/standard/machine/da540e0a/video.mp4 -o output.mp4

"""


import argparse
import os
import sys
import time
import tensorflow as tf
import cv2
import numpy as np
import warnings
import pandas as pd
from glob import glob
import mimetypes
# import validators
from utils import label_map_util
from utils import visual_utils as viz_utils
warnings.filterwarnings('ignore')  

category_index = None
detect_fn = None

def get_batch_args():
    ##Argumentparser is used to run through the terminal 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_directory',default='data/images',
                        help='image folder to be precessed')
    parser.add_argument('-o', dest='output_directory',default='data/output',
                        help='directory path for output images')
    parser.add_argument('-t', dest='score_thresh',default=0.5,
                        help='directory path for output images')
    args = parser.parse_args()
    return args

def VideoRecInit(WIDTH,HEIGHT,path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(path, fourcc, 30.0, (WIDTH,HEIGHT))
    return videowriter

def process_init(path_to_ckpt='exported_model/saved_model',
    path_to_labels='inputs/label_map.pbtxt'):
    
    global category_index, detect_fn

    start_time = time.time()
    print('Loading model...', end='\n')
    print ('Frozen Graph Path: ',path_to_ckpt)
    print ('Labels Path: ',path_to_labels)
    print ('-----------------------------------------------------------------------')

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(path_to_ckpt)
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels,use_display_name=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    
    return None

def sample_compute(_input,score_th=0.5):
    if isinstance(_input, str):
        print("Reading: " + _input)    
        image = cv2.imread(_input)
    else:
        image = _input
        print("Loading image/frame ...")

    image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    print(image_np.shape)
    img_height,img_width,_ = image_np.shape
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #print(detections['detection_classes'])
    image,bboxes = viz_utils.get_or_visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=3,
        min_score_thresh=score_th)

    return image,bboxes

def batch_compute(args):
    print ('-----------------------------------------------------------------------')
    print ('Tensorflow Version: ',tf.__version__)
    print ('Input Directory: ',args.input_directory)
    print ('Output Directory: ',args.output_directory)
    print ('Confidence Threshold: ', args.score_thresh)
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)
    print ('-----------------------------------------------------------------------')

    total_samples = len(os.listdir(args.input_directory))

    start_time = time.time()
    sample_no = 1
    for filename in os.listdir(args.input_directory):
        print ('processing image: ' + str(sample_no) + '/' + str(total_samples))
        sample_path = os.path.join(args.input_directory,filename)
        sample_out_path = os.path.join(args.output_directory,filename)
        image = sample_compute(sample_path,args.score_thresh)
        cv2.imwrite(sample_out_path,image)
        sample_no += 1
        print ('-----------------------------------------------------------------------')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return None

def video_compute(args):

    print ('-----------------------------------------------------------------------')
    print ('Tensorflow Version: ',tf.__version__)
    print ('Input Video: ',args.input_directory)
    print ('Output Video: ',args.output_directory)
    print ('Confidence Threshold: ', args.score_thresh)
    print ('-----------------------------------------------------------------------')

    ##initialize the video reader and writer
    cap = cv2.VideoCapture(args.input_directory)
    flag, frame = cap.read()
    if not flag:
        raise ValueError("Video Initialization Failed. Please make sure video path is valid.")

    (ht,wd,_) = frame.shape
    videowriter = VideoRecInit(wd,ht,args.output_directory)
    df = pd.DataFrame(columns=['frame_no','fX','fY','cx', 'cy','xmin','ymin','xmax','ymax','confidence'])

    start_id = 2000
    end_id = 4000

    frame_no = 1
    while frame_no < end_id:
        frame_no += 1
        if frame_no < 2000:
            continue
        flag, frame = cap.read()
        if flag == False:
            break

        #if frame_no % 10 != 0:
        #    continue
        print ('frame_no: ' + str(frame_no))
        timer = cv2.getTickCount()
        width_chunk = int(frame.shape[1] / 3)
        chunk_list = []
        for i in range(3):
            if i == 0:
                crop_img = frame[:, 0:width_chunk]
            else:
                crop_img = frame[:, width_chunk: width_chunk + width_chunk]
                width_chunk += width_chunk               
            out_frame,bboxes = sample_compute(crop_img,score_th=args.score_thresh)
            print("bboxes: ", bboxes)
            chunk_list.append(out_frame)
        if not bboxes:
            df = df.append({'frame_no':frame_no,'fX':wd,'fY':ht,'cx':0, 'cy':0,'xmin':0,'ymin':0,'xmax':0,'ymax':0,'confidence':0},ignore_index=True)
        else:
            centers = None
            for bbox in bboxes:
                df = df.append({'frame_no':frame_no,'fX':wd,'fY':ht,'cx':bbox[4], 'cy':bbox[5],'xmin':bbox[0],'ymin':bbox[1],'xmax':bbox[2],'ymax':bbox[3],'confidence':bbox[6]},ignore_index=True)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        print("fps : " + str(fps))
        final_image = np.hstack((chunk_list[0],chunk_list[1], chunk_list[2]))

        videowriter.write(final_image)
        df.to_csv(args.output_directory + '.csv')
    return None

if __name__ == '__main__':
    process_init()
    args = get_batch_args()
    if os.path.isdir(args.input_directory):
        batch_compute(args)
    elif os.path.isfile(args.input_directory):
        if mimetypes.guess_type(args.input_directory)[0].startswith('video'):
            video_compute(args)
        else:
            # frame = args.input_directory
            frame = cv2.imread(args.input_directory)
            width_chunk = int(frame.shape[1] / 3)
            chunk_list = []
            bboxes_dict = {}
            for i in range(3):
                if i == 0:
                    crop_img = frame[:, 0:width_chunk]
                else:
                    crop_img = frame[:, width_chunk: width_chunk + width_chunk]
                    width_chunk += width_chunk               
                # out_frame,bboxes = sample_compute(frame,score_th=args.score_thresh)
                out_frame, bboxes = sample_compute(crop_img,args.score_thresh)
                if bboxes:
                    bboxes_dict[i] = bboxes[0]
                cv2.imshow("image",out_frame)
                cv2.waitKey(0)
                print("bboxes: ", bboxes)
                chunk_list.append(out_frame)
            print("bboxes_dict: ", bboxes_dict)
            # print(bboxes)
            # print("type:", type(bboxes))
            cv2.imwrite('result_2.png',np.hstack((chunk_list[0],chunk_list[1], chunk_list[2])))
    elif validators.url(args.input_directory):
        video_compute(args)
    else:
        print('Please provide a valid image directory/file path')
