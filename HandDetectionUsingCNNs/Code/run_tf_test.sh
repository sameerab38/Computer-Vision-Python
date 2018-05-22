#!/bin/bash

#SBATCH --job-name egoHand
#SBATCH --qos=csqos
#SBATCH --partition=CS_q
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --error=/scratch/yli44/Datasets/egoHand-%j.error
#SBATCH --output=/scratch/yli44/Datasets/egoHand-%j.output
#SBATCH --mail-user=yli44@gmu.edu
#SBATCH --mail-type=BEGIN,FAIL,END

# Load the relevant modules needed for the job
module load hdf5_18/1.8.12
module load openblas/dynamic/0.2.8
module load cuda70/toolkit/7.0.28
module load boost/1_60_0/nompi
module load protobuf/3.5.1
module load OpenCV/2.4.10
module load python/python2.7.12
module load cuda70/blas/7.0.28
module load tensorflow/gpu/1.5.0
module load use.own
module load gcl-VER

module list

echo $CUDA_VISIBLE_DEVICES
python --version
# Start the job
#./tools/train_net.py --gpu 0 --weights data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel --imdb dota_train --cfg experiments/cfgs/cfg.yml --solver models/dota/VGG16/faster_rcnn_end2end/solver.prototxt --iter 0

#python /scratch/yli44/work/xView/baseline/inference/my_create_detections.py '/scratch/yli44/Datasets/xView/val_images/' -c '/scratch/yli44/work/xView/models_release/multires.pb' 
#python data_utilities/cut_out_objects.py

#/scratch/yli44/work/SegNet-Tutorial/caffe-segnet/build/tools/caffe train -gpu 0 -solver /scratch/yli44/work/SegNet-Tutorial/Models/segnet_solver.prototxt -weights /scratch/yli44/work/SegNet-Tutorial/Models/VGG_ILSVRC_16_layers.caffemodel
#python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
python research/object_detection/hand_detect.py