# Object_detection_Ova
A sample project to detect the custom object using Tensorflow object detection API

## Folder Structure
- Object_detection_Ova
  - src
    - label_conversion.py
	- data_cleanup.py
    - datasplitte_Subdirectories.py
    - data_augmentation.py 
    - data_aug 
		- __init__.py 
		- bbox_util.py	
		- data_aug.py	
    - xml_to_csv.py 
    - generate_tfrecord.py
  - pre_trained_models
    - *downloaded files for the choosen pre-trained model will come here* 
  - data
    - originalimages
      - *Annotations for your images will come here*
      - *all of your images will come here* 
    - augmentedImages/LabeledSplittedImages
	  - Test 
        - *all your images for testing will come here*	  
	  - Train 
		- *Annotations for your training images will come here*
		- *all of your images for training will come here*	  
	  - Validation
        - *Annotations for your validation images will come here*
        - *all of your images for validation will come here*
    - lable.pbtxt
    - train.record
	- Validation.record
   - Models
     - *inference graph of the trained model will be saved here*
   - training
     - *checkpoints of the trained model will be saved here*
   - config
     - *config file for the choosen model will come here*
   -testimages
    -result of testing will come here*
   - extra
    	- object_detection_tutorial_Ova.py   
    	- object_detection_tutorial_WebCam.py
	
## Setup Tensorflow models repository 
Now it's time when we will start using Tensorflow object detection API so download Tensorflow Object Detection API from:

https://github.com/tensorflow/models

Once you download the model repository, follow the insllation instruction in Tensorflow Object Detection API tutorial:
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

Now your Environment is all set to use Tensorlow object detection API.


## Convert the data to Tensorflow record format
In order to use Tensorflow API, you need to feed data in Tensorflow record format. I provide all the necessary scripts to help you to convert Pascal VOC format dataset to Tensorflow record format. 
You need to create a file for label map, in this repo it is *label.pbtxt*, with the dictionary of the label and the id of objects. Check *label.pbtxt* given in the repository to undestand the format, it's pretty simple 
(Note: name of the label should be same as what you gave while labeling object using the labelImg). Please update label_conversion.py simillar to label.pbtxt.

Now is time to create record file. From Object_detection_Ova as present working directory run the following command to create Tensorflow record:

To clean the annotation file or add information before using the data run data_cleanup.py: 
 - python  data_cleanup.py --data_path=data\originalimages\All

To split the data run datasplitte_Subdirectories.py
 - python src\datasplitte_Subdirectories.py --filename_extension=".png" --phases_info="percent,Train:80,Test:10,Validation:10" --data_path=data\originalimages\All --output_path=data\LabeledSplittedImages --ignored_directories="All_300x300"

If data augmentation is needed then run data_augmentation.py
 - python  src\data_cleanup.py --data_path=data\LabeledSplittedImages --output_path=data\augmentedImages

After preparing the data we need to convert the data to tenserflow record. Do the following:
 - python src\xml_to_csv.py  --data_path=data\LabeledSplittedImages --output_path=data
 - python src\generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
 - python src\generate_tfrecord.py --csv_input=data/Validation_labels.csv --output_path=data/Validation.record


## Training
Now that we have data in the right format to feed, we can go ahead with training our model. The first thing you need to do is to select the pre-trained model you would like to use. 
You could check and download a pret-rained model from [Tensorflow detection model zoo Github page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
Once downloaded, extract all file to the folder you had created for saving the pre-trained model files. Next, need to copy *models/research/sample/configs/<your_model_name.config>* and paste it in the project repo. 
You need to configure 5 paths in this file. Just open this file and search for PATH_TO_BE_CONFIGURED and replace it with the required path. I used pre-trained faster RCNN trained on COCO dataset and I have added 
modified config file (along with PATH_TO_BE_CONFIGURED as comment above lines which has been modified) for same in this repo. You could also play with other hyperparameters if you want. Now you are all set to train 
your model, just run th following command with models/research as present working directory

python object_detection/legacy/train.py --logtostderr --train_dir=<path_to_the folder_for_saving_checkpoints> --pipeline_config_path=<path_to_config_file>


An example is:

python object_detection/legacy/train.py --logtostderr --train_dir=D:\Object_detection_Ova\training\ --pipeline_config_path=D:\Object_detection_Ova\config\ssd_mobilenet_v1_coco.config

Checkpoints will be saved in training folder. 


## generate inference graph from saved checkpoints
python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=<path_to_config_file> --trained_checkpoint_prefix=<path to saved checkpoint> --output_directory=<path_to_the_folder_for_saving_inference_graph>

An example is:

python object_detection/export_inference_graph.py \    --input_type image_tensor \    --pipeline_config_path D:/Object_detection_ova/config/ssd_mobilenet_v1_coco.config \    --trained_checkpoint_prefix D:/Object_detection_ova/training/model.ckpt-5923 \    --output_directory D:/Object_detection_ova/Models


## Test the trained model
In order to test the re-tran model, All you need to do is to copy model/research/object_detection/object_detection_tutorial.ipynb and modify it to work with you inference graph. I modified the script and I have placed same in this repository inside the folder named as *extra*
A modified file is already given as object_detection_tutorial_Ova.py with this repo, you just need to change the path to model and path to images. The 2nd version of the file given as object_detection_tutorial_WebCam.py with this repo, uses Your Webcam to do object detection.




