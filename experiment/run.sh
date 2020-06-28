cd ..

# TODO: check every time
rm -rf checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn_t4_roi_t4_g1_lr02_2e4
# train, rpn_roi, kl_kl
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_dt_w_gtroi_rpn_roi.py --num-gpus 4 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml \
        --path_t checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --kd_T 4 \
        OUTPUT_DIR "checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn_10t4_roi_100t4_g1_lr02_2e4" \
        SOLVER.IMS_PER_BATCH "16" SOLVER.BASE_LR "0.02" SOLVER.CHECKPOINT_PERIOD "1000"\
        SOLVER.MAX_ITER "25000" \
        # SOLVER.STEPS "(33334, 44445)" SOLVER.WARMUP_ITERS "278"

rm -rf ~/Tensorboard
cp -r ./checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn_10t4_roi_100t4_g1_lr02_2e4/ ~/Tensorboard

# test to check overfitting
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --num-gpus 4 \
        --config-file checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_rpn_10t4_roi_100t4_g1_lr02_2e4/config.yaml \
        --eval-all --start-iter 14999
