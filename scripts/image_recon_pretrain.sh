python train.py \
    --gpus=1 \
    --resume /pasteur/u/esui/data/lafite/pre-trained-google-cc-best-fid.pkl \
    --outdir=/pasteur/u/esui/data/lafite/ckpt/ \
    --cfg_path configs/general_cfg.yaml \
    --temp=0.5 \
    --itd=10 \
    --itc=10 \
    --gamma=10 \
    --mirror=1 \
    --mixing_prob=1.0 \
    --normalize_prefix \
    --remove_mean \
    --add_gaussian_noise
    
# mixing_prob==1 -> no text data used \