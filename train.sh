

#nohup python -u train_densenet.py --data-dir /data/deeplearning/dataset/arrow --data-type kd --depth 169 --batch-size 512 --growth-rate 32 --drop-out 0 --reduction 0.5 --gpus=0,1,2,3 --train_prefix=train_0301 --val_prefix=val_0301 --model_prefix=/data/deeplearning/dataset/arrow/model_0301/densenet --retrain --model-load-epoch=2000 2>&1 | tee ./log/train_20180301.log &

python -u train_densenet.py --data-dir /data/deeplearning/dataset/arrow --data-type kd --depth 169 --batch-size 512 --growth-rate 32 --drop-out 0 --reduction 0.5 --gpus=0,1,2,3 --train_prefix=train_0301 --val_prefix=val_0301 --model_prefix=/data/deeplearning/dataset/arrow/model_0301/densenet --retrain --model-load-epoch=2000


