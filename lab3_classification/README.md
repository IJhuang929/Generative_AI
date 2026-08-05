第五步：推論（inference.py）

輸入：text（評審句子）+ 自行 parsing 的論文段落
輸出：讓 LLM 預測類別名稱，再 mapping 到 class id（0~4）
不可用 logit 分類，必須讓模型生成文字再對應


wandb 觀察清單（依優先序）
### 訓練動態（loss curves panel）
1. train/loss vs eval/loss — gap 若一路擴大 → overfit；可加 lora_dropout、降 epochs、減 r
2. eval/loss 是否單調下降 — 若第 2 個 epoch 開始反彈 → 已過擬合，最佳 checkpoint 是第 1 個 epoch
3. train/learning_rate — 確認 warmup 曲線符合預期（前 warmup_steps 線性上升，之後 linear decay 到 0）
4. train/grad_norm — 應在 0.5–5 區間穩定波動。若一直撞 max_grad_norm=1.0 → lr 太大；若 < 0.1 → 學習量不足，可加 lr 或 r
5. train/epoch 與 train/global_step — 對照 config 的 total_train_steps 確認跑完
### Dev 指標（每 epoch 一次）
6. dev/macro_f1 — 主指標。注意是否在最後一個 epoch 才達到峰值（可能還沒收斂）或中途達到（後面就是噪音）
7. dev/class_f1/Temporal 與 Overgeneralization — 兩個少數類，若遠低於 macro 平均 → class weights 沒生效或樣本太少
8. dev/class_precision/* vs dev/class_recall/* — recall 低 = 漏判（model 不敢選這類）；precision 低 = 誤判（過度自信）。決定要調 class weight 還是 prompt
9. dev/parse_failure_rate — 應接近 0；> 5% 表示 model 沒學會輸出規定格式（檢查 chat template 或加長 max_new_tokens）
10. dev/confusion_matrix_normalized — 看哪兩類最容易互相混淆（常見：Entity ↔ Number、Attribution ↔ Overgeneralization），決定要不要在 prompt 加區辨範例
### 資源 / 健全性
11. dev/gpu_peak_mem_gb vs config gpu_mem_gb — 還剩多少餘裕。若 < 80% → 可加大 batch_size 或 max_length；若 > 95% → 隨時會 OOM
12. Config 面板的 class_counts_train / class_weights — 確認分布與權重相符
13. Config 面板的 trainable_pct — 通常 LoRA r=16 約 0.5–2%，若異常高低代表 target_modules 沒套到對的層
14. Config 面板的 total_train_steps — 太少（< 50）→ 加 epochs；太多（> 2000）且 loss 早收斂 → 浪費算力
### Run 比較（多次 run 之間）
15. 同 project 多 run 的 dev/macro_f1 vs r / lr / lora_alpha 關係 — 用 wandb Sweeps 或 parallel coords plot 找最佳區域
16. train_seconds — 同樣 epoch 下不同設定的 wall-clock，搭配 F1 評估 cost-effectiveness


# inference.py command
python inference.py
# train.py command
python train.py \
    --data_dir . \
    --output_dir ./adapter_checkpoint \
    --max_length 512 \
    --evidence_max_chars 1000 \
    --r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lr 2e-4 \
    --batch_size 1 \
    --grad_accum 16 \
    --epochs 3 \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --no_class_weights \
    --lr_patience 1 \
    --lr_factor 0.5 \
    --wandb_project "hallucination_project" \
    --wandb_run_name "run_with_custom_trainer" \
    
    
    Unsloth 2026.4.8 patched 36 layers with 0 QKV layers, 0 O layers and 0 MLP layers.