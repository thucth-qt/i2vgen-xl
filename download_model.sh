# WEIGHTS_DIR="/data/thucth/weights/t2vgen"
WEIGHTS_DIR="weights"
mkdir -p $WEIGHTS_DIR

wget "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true" -P $WEIGHTS_DIR
mv $WEIGHTS_DIR/open_clip_pytorch_model.bin?download=true $WEIGHTS_DIR/open_clip_pytorch_model.bin

wget "https://huggingface.co/ali-vilab/i2vgen-xl/resolve/main/models/stable_diffusion_image_key_temporal_attention_x1.json?download=true" -P $WEIGHTS_DIR
mv $WEIGHTS_DIR/stable_diffusion_image_key_temporal_attention_x1.json?download=true $WEIGHTS_DIR/stable_diffusion_image_key_temporal_attention_x1.json

wget "https://huggingface.co/ali-vilab/i2vgen-xl/resolve/main/models/i2vgen_xl_00854500.pth?download=true" -P $WEIGHTS_DIR
mv $WEIGHTS_DIR/i2vgen_xl_00854500.pth?download=true $WEIGHTS_DIR/i2vgen_xl_00854500.pth

wget "https://huggingface.co/ali-vilab/i2vgen-xl/resolve/main/models/v2-1_512-ema-pruned.ckpt?download=true" -P $WEIGHTS_DIR
mv $WEIGHTS_DIR/v2-1_512-ema-pruned.ckpt?download=true $WEIGHTS_DIR/v2-1_512-ema-pruned.ckpt

wget "https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis/resolve/main/text2video_pytorch_model.pth?download=true" -P $WEIGHTS_DIR
mv $WEIGHTS_DIR/text2video_pytorch_model.pth?download=true $WEIGHTS_DIR/model_scope_1.pth

wget "https://huggingface.co/ali-vilab/i2vgen-xl/resolve/main/models/stable_diffusion_image_key_temporal_attention_x1.json?download=true" -P $WEIGHTS_DIR
mv $WEIGHTS_DIR/stable_diffusion_image_key_temporal_attention_x1.json?download=true $WEIGHTS_DIR/stable_diffusion_image_key_temporal_attention_x1.json