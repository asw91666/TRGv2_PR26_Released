TORCH_COMPILE_DISABLE=1 python trgv2_demo_video.py --video_path ./example/demo.mp4 \
  --frame_rate 30 \
  --fiter_threshold 0.9 \
  --do_render \
  --model_path checkpoint/TRG_PR26/checkpoint-30/state_dict.bin

