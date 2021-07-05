CUDA_VISIBLE_DEVICES=2 python tools/test_net.py --num-gpus 1 \
        --config-file checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_roi_g1_t4_lr003_5e4/config.yaml \
        --eval-all --dist-url "auto" --start-iter 25999