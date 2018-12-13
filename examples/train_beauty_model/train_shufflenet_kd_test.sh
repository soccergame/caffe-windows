#!/usr/bin/env sh
TOOLS=/home/18095062/caffe-master/.build_release/tools
#mkdir log
GLOG_logtostderr=0 GLOG_alsologtostderr=1 $TOOLS/caffe train  --gpu=6 --solver=/home/BRC_Project/dataset/Face/recognition/data/128_image/model/solver.prototxt --snapshot=/home/BRC_Project/dataset/Face/recognition/data/128_image/model/insightface_kd_fast_kd_shufflenetv1/shufflenet_v1_0.0001_iter_5537.solverstate --log_dir=./test_log 
#--engine=MKL2017 
#--weights=./shufflenet_am_iter_655000.caffemodel
#--snapshot=./shufflenet_FaceEthnic_Finetune_add_patch_crop_lr_0.01_han_part_and_uighur_iter_4000.caffemodel
#--gpu=0,1,2,3,4,5,6,7 --log_dir=./log --weights=./shufflenet_iter_7184.caffemodel
#--weights=./shufflenet_am_iter_655000.caffemodel
#--snapshot=./shufflenet_FaceGender_Finetune_FromAM_lr_0.001_iter_8394.solverstate
#--snapshot=./shufflenet_FaceEthnic_Finetune_add_patch_nocrop_lr_0.01_lmdb_iter_19691.solverstate
#,../ShuffleNet_MTL_Ethnic/shufflenet_FaceEthnic_Finetune_classifier_iter_68000.caffemodel,../ShuffleNet_MTL_Gender/shufflenet_FaceGender_Finetune_classifer_iter_144000.caffemodel,../ShuffleNet_MTL_Glasses/shufflenet_FaceGlasses_Finetune_classifer_iter_344000.caffemodel
#--weights=../ShuffleNet_MTL_Age/shufflenet_FaceAge_MultiOutput_merge_head_tail_47_classes_SampleID_lmdb_Finetune_Classifier_lr_1e-5_iter_200000.caffemodel,../ShuffleNet_MTL_Ethnic/shufflenet_FaceEthnic_Finetune_classifier_iter_68000.caffemodel,../ShuffleNet_MTL_Gender/shufflenet_FaceGender_Finetune_classifer_iter_144000.caffemodel,../ShuffleNet_MTL_Glasses/shufflenet_FaceGlasses_Finetune_classifer_iter_344000.caffemodel,../ShuffleNet_MTL_Emotion/shufflenet_FaceEmotion_Finetune_classifier_iter_396000.caffemodel
#--weights=./shufflenet_FaceAtrributes_and_emotion_lr_0.0001_iter_320000.caffemodel
#
#--snapshot=./shufflenet_FaceAtrributes_and_emotion_lr_0.0001_iter_324667.solverstate
#--snapshot=./shufflenet_FaceAtrributes_and_emotion_adjust_param_lr_0.0001_iter_116344.solverstate