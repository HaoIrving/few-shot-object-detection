cd ..

# g1_t4_lr002
CUDA_VISIBLE_DEVICES=0 python tools/train_distillwogtroi.py --num-gpus 1 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml \
        --path_t checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --kd_T 4 \
        OUTPUT_DIR "checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_distillwogtroi_g1_t4_lr002" \
        SOLVER.IMS_PER_BATCH "16" SOLVER.BASE_LR "0.0001" 