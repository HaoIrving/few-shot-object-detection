cd ..

CUDA_VISIBLE_DEVICES=2,3 python tools/test_net.py --num-gpus 2 \
        --config-file checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1_dt_w_gtroi_roi_g1_t4_lr003/config.yaml \
        --eval-all --dist-url "auto"