import boto3
import re
import os
import wget
import time
from time import gmtime, strftime
import sys
import json

# Different algorithms have different registry and account parameters
# see: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/amazon_estimator.py#L272
def get_image_uri(region_name):
    """Return object detection algorithm image URI for the given AWS region"""
    account_id = {
        "us-east-1": "811284229777",
        "us-east-2": "825641698319",
        "us-west-2": "433757028032",
        "eu-west-1": "685385470294",
        "eu-central-1": "813361260812",
        "ap-northeast-1": "501404015308",
        "ap-northeast-2": "306986355934",
        "ap-southeast-2": "544295431143",
        "us-gov-west-1": "226302683700"
    }[region_name]
    return '{}.dkr.ecr.{}.amazonaws.com/object-detection:latest'.format(account_id, region_name)

start = time.time()

region_name = sys.argv[1]
role = sys.argv[2]
bucket = sys.argv[3]
stack_name = sys.argv[4]
commit_id = sys.argv[5]
commit_id = commit_id[0:7]

training_image = get_image_uri(region_name)
timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        wget.download(url, filename)


def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)

# TODO: Load data and split in to train/test split

# Download the training zip file
input_s3_path="s3://sagemaker-objdetect/rego-plate-detection/AustralianNumberPlate.zip"
bucket = 'sagemaker-objdetect' # custom bucket name.
prefix = 'rego-plate-detection'

# Download and unzip the file into temp folder
tmp_dir="tmp_objectdetection_cs"
local_input_file = "{}/data.zip".format(tmp_dir)
local_data_dir = "./{}".format(tmp_dir)
!mkdir -p $tmp_dir

!aws s3 cp $input_s3_path $local_input_file
!echo $local_data_dir
!unzip  -o  $local_input_file -d $local_data_dir

input_raw_data_dir="{}/Selected".format(local_data_dir)
input_processed_data_dir="{}/data".format(tmp_dir)
!mkdir -p $input_processed_data_dir

import xml.etree.ElementTree as ET
from pathlib import Path
import os
import json

def transform(input_xml, images_dir, class_index, output_dir):
    formatted_annotation = {}
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # Image file
    formatted_annotation['file']  =  root.find("path").text.split("\\")[-1]
    
    #Image size
    img_size_w =root.find("size/width").text
    img_size_h =root.find("size/height").text
    img_size_d =root.find("size/depth").text
    formatted_annotation['image_size']= [{"width": img_size_w, "height": img_size_h, "depth": img_size_d }]
    
    #Image annotations
    formatted_annotation['annotations'] = []
    formatted_annotation['categories'] = []
    categories = {}
    for e in  root.findall("object"):
        annotation = {}
        class_name = e.find('name').text
        annotation['class_id'] = class_index[class_name]
        annotation['left'] = int(e.find('bndbox/xmin').text)
        annotation['width'] = int(e.find('bndbox/xmax').text) - int(e.find('bndbox/xmin').text) 
        annotation['top'] = int(e.find('bndbox/ymin').text) 
        annotation['height'] = int(e.find('bndbox/ymax').text) - int(e.find('bndbox/ymin').text) 

        # Add annotation
        formatted_annotation['annotations'].append(annotation)
        
        # Add categories list , only unique
        if not class_name in categories:
            formatted_annotation['categories'].append({ 'class_id':  class_index[class_name], 'name': class_name})
    
    # Write the output file using the same name as the image file, but with .json extension
    file_name =  "{}.json".format(formatted_annotation['file'].split(".")[-2:-1][0])
    with open(os.path.join(output_dir, file_name) ,"w") as f   :
        f.write(json.dumps(formatted_annotation))
    print(formatted_annotation)
    
    ## Move the image file to the same directory
    os.rename(os.path.join(input_raw_data_dir, formatted_annotation['file']), os.path.join(output_dir, formatted_annotation['file']))

labels_index={'NumberPlate':0}

# Transform the data into the expected format
import os
files_list = os.listdir(input_raw_data_dir)
for f in files_list:
    # xmlinputfile has no extension
    if not "." in f:
        transform(os.path.join(input_raw_data_dir, f), input_raw_data_dir,labels_index , input_processed_data_dir)
   
      
# Split into train and validation

train_dir="{}/train".format(tmp_dir)
train_annot_dir="{}/train_annotation".format(tmp_dir)
validation_dir="{}/validation".format(tmp_dir)
validation_annot_dir="{}/validation_annotation".format(tmp_dir)


#Create folders to store the data and annotation files
!mkdir -p $train_dir $train_annot_dir $validation_dir $validation_annot_dir



import os
import json
import glob
jsons = glob.glob('{}/*.json'.format(input_processed_data_dir))

train_size= int(len(jsons)*.8)


import shutil
from sklearn.model_selection import train_test_split, learning_curve
train_jsons, val_jsons = train_test_split(jsons, test_size = 0.2, random_state = 777, shuffle=True)



#Moving training files to the training folders
for f in train_jsons:
    image_file =  f.split('.')[0] + '.jpg'
    shutil.move(image_file, train_dir)
    shutil.move(f, train_annot_dir)

#Moving validation files to the validation folders
for f in val_jsons:
    image_file =  f.split('.')[0]+'.jpg'
    shutil.move(image_file, validation_dir)
    shutil.move(f, validation_annot_dir)

    
train_channel = prefix + '/train'
validation_channel = prefix + '/validation'
train_annotation_channel = prefix + '/train_annotation'
validation_annotation_channel = prefix + '/validation_annotation'

sess.upload_data(path=train_dir, bucket=bucket, key_prefix=train_channel)
sess.upload_data(path=validation_dir, bucket=bucket, key_prefix=validation_channel)
sess.upload_data(path=train_annot_dir, bucket=bucket, key_prefix=train_annotation_channel)
sess.upload_data(path=validation_annot_dir, bucket=bucket, key_prefix=validation_annotation_channel)

s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)
s3_train_annotation = 's3://{}/{}'.format(bucket, train_annotation_channel)
s3_validation_annotation = 's3://{}/{}'.format(bucket, validation_annotation_channel)

s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)

from sagemaker.amazon.amazon_estimator import get_image_uri

print ("Setting Algorithm Settings")

# Specify the base network
base_network = 'resnet-50'
# For this training, we will use 18 layers
label_width = 600
# we need to specify the input image shape for the training data
image_shape = 512
# we also need to specify the number of training samples in the training set
# for caltech it is 15420
num_training_samples = train_size
# specify the number of output classes
num_classes = 1
# batch size for training
mini_batch_size = 16
# number of epochs
epochs = 1
# learning rate
learning_rate = 0.001

# create unique job name
job_name = stack_name + "-" + commit_id + "-" + timestamp
training_params = \
{
    # specify the training docker image
    "AlgorithmSpecification": {
        "TrainingImage": training_image,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": 's3://{}'.format(bucket)
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.p2.8xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        "base_network": base_network,
        "use_pretrained_model": "1",
        "num_classes": str(num_classes),
        "mini_batch_size": str(mini_batch_size),
        "epochs": str(epochs),
        "learning_rate": str(learning_rate),
        "lr_scheduler_step": str(10),
        "lr_scheduler_factor": str(0.1),
        "optimizer": "sgd",
        "momentum": str(0.9),
        "weight_decay": str(0.0005),
        "overlap_threshold": str(0.5),
        "nms_threshold": str(0.45),
        "image_shape": str(image_shape),
        "label_width": str(label_width),
        "num_training_samples": str(train_size)
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 360000
    },
#Training data should be inside a subdirectory called "train" and "train_annotation"
#Validation data should be inside a subdirectory called "validation" and "validation_annotation"
#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": 's3://sagemaker-objdetect/rego-plate-detection/train/'.format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "image/jpeg",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": 's3://sagemaker-objdetect/rego-plate-detection/validation/'.format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "image/jpeg",
            "CompressionType": "None"
        },
        {
            "ChannelName": "train_annotation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": 's3://sagemaker-objdetect/rego-plate-detection/train_annotation/'.format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "image/jpeg",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation_annotation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": 's3://sagemaker-objdetect/rego-plate-detection/validation_annotation/'.format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "image/jpeg",
            "CompressionType": "None"
        }
    ]
}
print('Training job name: {}'.format(job_name))
print('\nInput Data Location: {}'.format(training_params['InputDataConfig'][0]['DataSource']['S3DataSource']))

# create the Amazon SageMaker training job
sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))


# creating configuration files so we can pass parameters to our sagemaker endpoint cloudformation

config_data_qa = {
  "Parameters":
    {
        "BucketName": bucket,
        "CommitID": commit_id,
        "Environment": "qa",
        "ParentStackName": stack_name,
        "SageMakerRole": role,
        "SageMakerImage": training_image,
        "Timestamp": timestamp
    }
}

config_data_prod = {
  "Parameters":
    {
        "BucketName": bucket,
        "CommitID": commit_id,
        "Environment": "prod",
        "ParentStackName": stack_name,
        "SageMakerRole": role,
        "SageMakerImage": training_image,
        "Timestamp": timestamp
    }
}


json_config_data_qa = json.dumps(config_data_qa)
json_config_data_prod = json.dumps(config_data_prod)

f = open( './CloudFormation/configuration_qa.json', 'w' )
f.write(json_config_data_qa)
f.close()

f = open( './CloudFormation/configuration_prod.json', 'w' )
f.write(json_config_data_prod)
f.close()

end = time.time()
print(end - start)
