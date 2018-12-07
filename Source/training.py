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

from sagemaker.amazon.amazon_estimator import get_image_uri

print ("Setting Algorithm Settings")

train_size = 336

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
        "InstanceType": "ml.p2.16xlarge",
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
