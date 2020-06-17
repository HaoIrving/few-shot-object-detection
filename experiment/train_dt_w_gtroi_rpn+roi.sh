cd ..

# TODO: check every time
rm -rf checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn+roi_g1_t4_lr01

# g1_t4_lr002
CUDA_VISIBLE_DEVICES=0,1 python tools/train_dt_w_gtroi_rpn+roi.py --num-gpus 2 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml \
        --path_t checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --kd_T 4 \
        OUTPUT_DIR "checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn+roi_g1_t4_lr005" \
        SOLVER.IMS_PER_BATCH "16" SOLVER.BASE_LR "0.005" SOLVER.CHECKPOINT_PERIOD "1000"

rm -rf ~/Tensorboard
cp -r ./checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn+roi_g1_t4_lr005/ ~/Tensorboard