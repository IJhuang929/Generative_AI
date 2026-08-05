# v2
add EMA, v_prediction + MinSNR 
# v3 
同時做三件事：fp16 + GradScaler、EMA on CPU、generate_and_save_images 的 eval 也用 fp16。 減少 VRAM 使用量
# v4
之後只要改 train_epochs = N，warmup 和 cosine 的步數會全部自動跟著算。UNet channels (256,512,512,512), down/up blocks AttnDownBlock2D