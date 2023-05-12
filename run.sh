config_files=(Uformer_multivarall_zscore_option.yml RCAN_multivarall_zscore_option.yml SwinIR_c180_6x6_multivarall_zscore_option.yml \
    Uformer_MultiFuseOut_multivarall_zscore_option.yml Uformer_MultiScaleHGT_MultiFuseOut_multivarall_zscore_option.yml \
    Uformer_MultiScaleHGT_multivarall_zscore_option.yml UNet_multivarall_zscore_option.yml)
for config_file in ${config_files[@]}
do
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --master_port=12316 train.py \
    -opt paper_options/${config_file} \
    --launcher pytorch
done
