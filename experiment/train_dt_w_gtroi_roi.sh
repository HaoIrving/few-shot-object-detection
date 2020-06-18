cd ..

# TODO: check every time
rm -rf checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_roi_g1_t4_lr003_3e4

# g1_t4_lr002
CUDA_VISIBLE_DEVICES=2,3 python tools/train_dt_w_gtroi_roi.py --num-gpus 2 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml \
        --path_t checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --kd_T 4 --dist-url "auto"\
        OUTPUT_DIR "checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_roi_g1_t4_lr003_3e4" \
        SOLVER.IMS_PER_BATCH "16" SOLVER.BASE_LR "0.003" SOLVER.CHECKPOINT_PERIOD "1000"\
        SOLVER.MAX_ITER "30000" SOLVER.STEPS "(20000, 26700)" SOLVER.WARMUP_ITERS "200"

# rm -rf ~/Tensorboard
# cp -r ./checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_roi_g1_t4_lr003/ ~/Tensorboard

bash experiment/test_dt_w_gtroi_roi.sh