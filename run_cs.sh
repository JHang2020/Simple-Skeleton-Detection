#### HiCLR NTU-60 xsub ####

# Pretext
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xsub_joint.yaml
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xsub_motion.yaml
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xsub_bone.yaml

# Linear_eval
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xsub_joint.yaml
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xsub_motion.yaml
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xsub_bone.yaml

#finetune
python main.py finetune_evaluation --config config/release/gcn_ntu60/finetune/xsub_joint.yaml
# Ensemble
python ensemble_ntu_cs.py
#detection

## sliding window TIP
python main.py detection_sw_evaluation --config /mnt/netdisk/zhangjh/Code/skeleton_detection/detection/config/release/gcn_ntu60/finetune/xsub_joint_sliding_window.yaml

## downsample ECCV 2018 + CVPRW 2016 
#https://github.com/google/graph_distillation
#https://github.com/imatge-upc/activitynet-2016-cvprw/blob/master/scripts/process_prediction.py#L158
python main.py detection_ds_evaluation --config /mnt/netdisk/zhangjh/Code/skeleton_detection/detection/config/release/gcn_ntu60/finetune/xsub_joint_sliding_window.yaml

## downsample linear
python main.py detection_ds_linear_evaluation --config /mnt/netdisk/zhangjh/Code/skeleton_detection/detection/config/release/gcn_ntu60/linear_eval/detection.yaml


#visualize
#python main.py vis_evaluation --config config/release/ntu60/finetune/xsub_motion.yaml