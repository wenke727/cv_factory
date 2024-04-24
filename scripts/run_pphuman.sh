OUTPUT_FOLDER=./data/mot

cd ../PaddleDetection;

python deploy/pipeline/pipeline.py \
    --config deploy/pipeline/config/infer_cfg_pphuman.yml \
    -o MOT.enable=True \
    MOT.model_dir=output_inference/mot_ppyoloe_l_36e_pipeline \
    --video_file=../cv_factory/$CUR_DIR$1 \
    --output_dir=../cv_factory/$OUTPUT_FOLDER

cd ../cv_factory
