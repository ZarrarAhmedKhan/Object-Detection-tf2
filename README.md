# Object-Detection-tf2

TF2 Object Detection - tensorflow==2.7.0

## Preprocessing

TensorFlow Object Detection API Installation

Inside inputs folder

* place "label_map.pbtxt" and "tf.record files"

Inside config folder

* download the models from here https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

* place the config file in it and change the paths plus parameters

Parent Directory

* git clone https://github.com/tensorflow/models.git

> `%cd models/research`

> `protoc object_detection/protos/*.proto --python_out=.`

> `cp object_detection/packages/tf2/setup.py .`

> `python3 -m pip install --use-feature=2020-resolver .`

Check whether installation done successfully

> `python3 object_detection/builders/model_builder_tf2_test.py`

Link the object detection folder inside the models with parent directory

> `ln -s models/research/object_detection .`

## Training

* Make sure the config file
* change paths in run_train

> `chmod +x run_train`

> `./run_train`

## Evaluation

* change paths in run_eval

> `chmod +x run_eval`

> `./run_eval`

## Exporting model

* change paths in run_train

> `chmod +x run_export`

> `./run_export`

## Inference

* chnage path of model and label_map.pbtxt

> `python3 inference.py -i path/video.mp4 -o path/res_video.mp4`

> `python3 inference.py -i path/image.jpg -o path/res_image.jpg`

## Errors

> ImportError: cannot import name '_registerMatType' from 'cv2.cv2' (/usr/local/lib/python3.7/dist-packages/cv2/cv2.cpython-37m-x86_64-linux-gnu.so)

mismatching of opencv-python and opencv-python-headless versions

So, for now it is

> `pip3 uninstall opencv-python-headless -y`

> `pip3 install opencv-python-headless==4.1.2.30`

## Suggestions

If you want to add how many max checkpoints you want to keep **checkpoint_max_to_keep** 

For Tensorflow Object API 2, in **model_main_tf2.py** line 104, change to this:

>         model_lib_v2.train_loop(
>
>         pipeline_config_path=FLAGS.pipeline_config_path,
>          
>         model_dir=FLAGS.model_dir,
>          
>         train_steps=FLAGS.num_train_steps,
>         
>         use_tpu=FLAGS.use_tpu,
>         
>         checkpoint_every_n=FLAGS.checkpoint_every_n,
>         
>         record_summaries=FLAGS.record_summaries,
>         
>         checkpoint_max_to_keep=500)
