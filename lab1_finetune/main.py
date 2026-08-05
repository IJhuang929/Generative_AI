import os
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import re
from huggingface_hub import login

# login(token="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")
# 使用自己的 Hugging Face token 來下載模型和上傳訓練好的權重，請替換成你自己的 token
login(token="YOUR_HF_TOKEN")

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH = "dataset/dataset.csv"
BENCHMARK_PATH = "dataset/benchmark.csv"
# 要train的話要開
DO_TRAIN = False
SYSTEM_PROMPT = "You are a board-certified pathology expert. You are taking a highly strict medical board exam. Carefully read the question and evaluate EVERY option. You MUST select exactly ONE correct option. Do not explain, just output the letter."
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}
# 要從checkpoint繼續訓練的話，填checkpoint路徑，沒有的話填 None 就會從頭開始訓練
RESUME_FROM = None
# 復現用的 checkpoint，裡面只有 LoRA 權重，trainer 會從 base model 開始訓練，但會載入這些 LoRA 權重來加速收斂
LORA_PATH_FOR_INFERENCE = "./saved_models/llama_20260329_161834/best_accuracy_checkpoint"


def get_manual_few_shot_examples(train_df, question_ids):
    """
    use question_id to select specific few-shot examples from the training set, ensuring they are truly in the training data and not accidentally from Val/Test splits. This prevents data leakage and ensures the integrity of the few-shot evaluation.
    """
    if "question_id" in train_df.columns:
        # check which question_ids are actually present in the training set
        valid_ids = [
            qid for qid in question_ids if qid in train_df["question_id"].values
        ]
        missing_ids = set(question_ids) - set(valid_ids)

        if missing_ids:
            print(
                f"Warning: missing IDs {missing_ids} not in training set (might be in Val/Test)! Automatically excluding to avoid data leakage."
            )

        few_shot_df = train_df[train_df["question_id"].isin(valid_ids)].copy()

        if not few_shot_df.empty:
            few_shot_df["question_id"] = pd.Categorical(
                few_shot_df["question_id"], categories=valid_ids, ordered=True
            )
            few_shot_df = few_shot_df.sort_values("question_id")

    else:
        # if no question_id column, fallback to index-based selection but still check validity
        valid_ids = [qid for qid in question_ids if qid in train_df.index]
        missing_ids = set(question_ids) - set(valid_ids)

        if missing_ids:
            print(
                f"Warning: missing Index {missing_ids} in train_df! Ignoring these IDs to avoid data leakage."
            )

        few_shot_df = train_df.loc[valid_ids].copy()

    if few_shot_df.empty:
        raise ValueError(
            "Fault: missing all requested IDs in training set! Please open train_split_view.csv and reselect."
        )

    return few_shot_df.reset_index(drop=True)


def build_few_shot_messages(question, options, answer=None, few_shot_df=None):
    """Build chat messages with optional few-shot examples prepended."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if few_shot_df is not None:
        for _, ex in few_shot_df.iterrows():
            ex_user = (
                f"Question: {ex['question']}\n"
                f"A. {ex['opa']}\nB. {ex['opb']}\nC. {ex['opc']}\nD. {ex['opd']}\n"
                "Answer with the single best letter."
            )
            messages.append({"role": "user", "content": ex_user})
            messages.append({"role": "assistant", "content": LABEL_MAP[ex["ans"]]})

    user_content = (
        f"Question: {question}\n"
        f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n"
        "Answer with the single best letter."
    )
    messages.append({"role": "user", "content": user_content})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def convert_to_mc_dataset(df, tokenizer, few_shot_df=None, is_train=False):
    mc_data = []
    for _, row in df.iterrows():
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        correct_idx = int(row["ans"])

        if is_train:
            # shuffle options during training to prevent overfitting to position, but keep track of the correct answer's new position
            order = list(range(4))
            random.shuffle(order)
            current_options = [options[i] for i in order]
            correct_answer = ["A", "B", "C", "D"][order.index(correct_idx)]
        else:
            # eval and inference use original order to ensure consistency
            current_options = options
            correct_answer = LABEL_MAP[row["ans"]]

        # generate full prompt with answer (for training) or without answer (for eval/inference), but in both cases we need the full text to calculate the correct masking length
        full_text = tokenizer.apply_chat_template(
            build_few_shot_messages(
                row["question"],
                current_options,
                answer=correct_answer,
                few_shot_df=few_shot_df,
            ),
            tokenize=False,
        )
        # generate prompt text without answer for both training and eval, since the model should learn to predict the answer token
        prompt_text = tokenizer.apply_chat_template(
            build_few_shot_messages(
                row["question"], current_options, answer=None, few_shot_df=few_shot_df
            ),
            tokenize=False,
            add_generation_prompt=True,
        )
        mc_data.append({"text": full_text, "prompt": prompt_text})

    return Dataset.from_list(mc_data)


class ValidationAccuracyCallback(TrainerCallback):
    def __init__(self, val_df, tokenizer, target_accuracy=0.9, log_dir="./runs"):
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.val_accuracies = []
        self.target_accuracy = target_accuracy
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.best_accuracy = 0.0
        self.best_accuracy_checkpoint = None
        self.trainer = None  # set after Trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.tb_writer.add_scalar("train/loss", logs["loss"], step)
        if "eval_loss" in logs:
            self.tb_writer.add_scalar("eval/loss", logs["eval_loss"], step)
        if "learning_rate" in logs:
            self.tb_writer.add_scalar(
                "train/learning_rate", logs["learning_rate"], step
            )
        self.tb_writer.flush()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.val_df is None:
            return

        model.eval()

        print(
            f"\nEvaluating Validation Accuracy (3-vote majority) on {len(self.val_df)} samples..."
        )
        questions = self.val_df["question"].tolist()
        options_list = [
            {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
            for _, row in self.val_df.iterrows()
        ]
        ans_list = self.val_df["ans"].tolist()

        preds = predict_by_generation(
            model, self.tokenizer, questions, options_list, n_votes=1, batch_size=8
        )
        correct = sum(pred == LABEL_MAP.get(ans) for pred, ans in zip(preds, ans_list))

        acc = correct / len(self.val_df)
        self.val_accuracies.append(acc)

        print(f"Epoch {state.epoch} - Validation Accuracy: {acc:.4f}\n")
        self.tb_writer.add_scalar("val/accuracy", acc, state.global_step)
        self.tb_writer.flush()

        # checkpoint with best accuracy
        if acc > self.best_accuracy and self.trainer is not None:
            self.best_accuracy = acc
            ckpt_path = f"{args.output_dir}/best_accuracy_checkpoint"
            self.trainer.save_model(ckpt_path)
            self.best_accuracy_checkpoint = ckpt_path
            print(f"New best accuracy {acc:.4f} — saved to {ckpt_path}")

        model.train()

        # control parameter for Early Stopping
        if acc >= self.target_accuracy:
            print(
                f"Validation accuracy reached {acc:.4f} (>= {self.target_accuracy:.4f})."
            )
            print("Stopping training early to prevent overfitting!")
            control.should_training_stop = True


def train_model(
    train_dataset,
    eval_dataset,
    tokenizer,
    val_df=None,
    resume_from=None,
    output_dir="./saved_models",
    log_dir="./runs",
    run_ts="",
):
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa"
    )

    model = prepare_model_for_kbit_training(model)

    if resume_from:
        print(f"Resuming from LoRA checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,           
            lora_alpha=32, 
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,  
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        disable_tqdm=False,
        output_dir=output_dir,
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=16,  
        per_device_eval_batch_size=1,       
        eval_accumulation_steps=1,
        num_train_epochs=3,  
        warmup_ratio=0.05,
        neftune_noise_alpha=5,
        label_smoothing_factor=0.05,
        learning_rate=2e-4,      
        weight_decay=0.05,
        logging_steps=10,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        eval_strategy="epoch",
        bf16=True,
        tf32=True,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",
        logging_dir=log_dir,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    callbacks = []
    acc_callback = None
    if val_df is not None:
        acc_callback = ValidationAccuracyCallback(
            val_df, tokenizer, target_accuracy=1.01, log_dir=log_dir
        )
        callbacks.append(acc_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    if acc_callback is not None:
        acc_callback.trainer = trainer  # for saving best accuracy checkpoint

    # best_accuracy_checkpoint only have LoRA weights，trainer start from scratch
    has_trainer_state = resume_from and os.path.exists(
        os.path.join(resume_from, "trainer_state.json")
    )
    trainer.train(resume_from_checkpoint=resume_from if has_trainer_state else None)
    final_step = trainer.state.global_step

    # plot
    train_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    loss_steps = [log["step"] for log in trainer.state.log_history if "loss" in log]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_steps, train_loss, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    if acc_callback and acc_callback.val_accuracies:
        epochs = list(range(1, len(acc_callback.val_accuracies) + 1))
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs,
            acc_callback.val_accuracies,
            label="Validation Accuracy",
            color="orange",
            marker="o",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Curve")
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plot_path = f"training_results_{run_ts}.png"
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")

    tb_writer = (
        acc_callback.tb_writer
        if acc_callback is not None
        else SummaryWriter(log_dir=log_dir)
    )
    best_ckpt = (
        acc_callback.best_accuracy_checkpoint if acc_callback is not None else None
    )
    return model, final_step, tb_writer, best_ckpt


def predict_by_generation(
    model,
    tokenizer,
    questions,
    options_list,
    batch_size,
    n_votes=1,
    temperature=0.4,
    few_shot_df=None,
):
    """
    Predict answers by generating text from the model.
    For each question, we generate n_votes times (with optional temperature sampling) and take a majority vote to improve robustness. The model's output is parsed to extract the predicted option letter (A/B/C/D). This method relies on the model's ability to generate coherent text that includes the correct answer letter, and the regex parsing ensures we can handle variations in the generated output.
    """

    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    option_keys = ["A", "B", "C", "D"]
    all_votes = [[] for _ in range(len(questions))]

    for _ in range(n_votes):
        prompts = []
        orders = []  
        for q, opts in zip(questions, options_list):
            original_opts = [opts[k] for k in option_keys]
            order = list(range(4))

            
            if n_votes > 1:
                random.shuffle(order)
            shuffled = [original_opts[i] for i in order]
            orders.append(order)

            
            messages = build_few_shot_messages(
                q, shuffled, answer=None, few_shot_df=few_shot_df
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        vote_results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                
                gen_kwargs = {
                    "max_new_tokens": 5,  
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if n_votes > 1:
                    gen_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": temperature,
                            "top_p": 0.9,
                        }
                    )
                else:
                    gen_kwargs.update({"do_sample": False})

                outputs = model.generate(**inputs, **gen_kwargs)

           
            input_lens = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_lens:]
            generated_texts = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            for text in generated_texts:
                
                match = re.search(r"[A-D]", text.upper())
                if match:
                    vote_results.append(match.group(0))
                else:
                    vote_results.append("A")  

        
        for j, pred_letter in enumerate(vote_results):
            idx = option_keys.index(pred_letter)
            original_idx = orders[j][idx]
            original_letter = option_keys[original_idx]
            all_votes[j].append(original_letter)
            
        torch.cuda.empty_cache()
    tokenizer.padding_side = original_padding_side
    return [Counter(votes).most_common(1)[0][0] for votes in all_votes]


def run_inference(model, tokenizer, df, batch_size, few_shot_df=None, isbenchmark=False):
    questions = df["question"].tolist()
    options = [
        {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        for _, r in df.iterrows()
    ]
    model.eval()
    if isbenchmark:
        return predict_by_generation(
            model,
            tokenizer,
            questions,
            options,
            batch_size=batch_size,
            n_votes=11,
            temperature=0.3,
            few_shot_df=few_shot_df
        )
    else:
        return predict_by_generation(
            model,
            tokenizer,
            questions,
            options,
            batch_size=batch_size,
            n_votes=3,
            temperature=0.4,
            few_shot_df=few_shot_df,
        )


def main():
    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./saved_models/llama_{RUN_TS}"
    log_dir = f"./runs/{RUN_TS}"
    print(f"Run timestamp: {RUN_TS}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"Using model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("-" * 50)
    print("Loading and spliting dataset...")
    df = pd.read_csv(DATA_PATH).dropna(subset=["ans"])
    df = df.drop_duplicates(subset=["question"])
    # 80% Train, 20% Temp (to be split into Val and Test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    # Temp split into 50% Val and 50% Test => 10% Val, 10% Test of original data
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("-" * 50)
    print(
        f"Dataset split: Train({len(train_df)}), Val({len(val_df)}), Inference({len(test_df)})"
    )

    best_example_ids = [4665, 953]         #[3177,4044], [4665, 7808], [953,2519]
    few_shot_df = get_manual_few_shot_examples(train_df, best_example_ids)
    print(f"Manual Few-shot examples selected (IDs: {best_example_ids}):")
    for _, ex in few_shot_df.iterrows():
        print(f"  Q: {ex['question'][:60]}...  A: {LABEL_MAP[ex['ans']]}")

    MAX_LENGTH = 256     

    print("-" * 50)
    print("Converting DataFrames to Hugging Face Datasets...")

    train_hf_dataset = convert_to_mc_dataset(
        train_df, tokenizer, few_shot_df=few_shot_df, is_train=True
    )
    val_hf_dataset = convert_to_mc_dataset(
        val_df, tokenizer, few_shot_df=few_shot_df, is_train=False
    )

    # Tokenize: mask everything before the assistant's answer token
    def tokenize_function(examples):
        full = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        prefix_lens = [
            len(tokenizer(p, truncation=True, max_length=MAX_LENGTH).input_ids)
            for p in examples["prompt"]
        ]
        labels = [ids[:] for ids in full["input_ids"]]
        for i, prefix_len in enumerate(prefix_lens):
            for j in range(len(labels[i])):
                if j < prefix_len or full["input_ids"][i][j] == tokenizer.pad_token_id:
                    labels[i][j] = -100
        full["labels"] = labels
        return full

    print("Pre-tokenizing datasets with multiple processes (num_proc=4)...")
    tokenized_train = train_hf_dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_val = val_hf_dataset.map(tokenize_function, batched=True, num_proc=4)

    
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    tokenized_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    final_step = 0
    tb_writer = SummaryWriter(log_dir=log_dir)
    if DO_TRAIN:
        print("-" * 50)
        print("Starting training...")
        model, final_step, tb_writer, best_ckpt = train_model(
            tokenized_train,
            tokenized_val,
            tokenizer,
            val_df,
            RESUME_FROM,
            output_dir=output_dir,
            log_dir=log_dir,
            run_ts=RUN_TS,
        )
        if best_ckpt and os.path.isdir(best_ckpt):
            print(f"Loading best accuracy checkpoint: {best_ckpt}")
            model.load_adapter(best_ckpt, adapter_name="best_accuracy")
            model.set_adapter("best_accuracy")
            model.eval()
    else:
        print("-" * 50)
        print("Skipping training. Loading base model for inference...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # 如果要重train要註解掉
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa"
        )
        print(f"Loading LoRA weights from: {LORA_PATH_FOR_INFERENCE}")
        model = PeftModel.from_pretrained(base_model, LORA_PATH_FOR_INFERENCE)
        # 有要train
        # model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
        model.eval()

    # inference on 10% test split
    print("-" * 50)
    print("Running inference on 10% test split...")
    test_df = test_df.copy()
    test_df["pred"] = run_inference(model, tokenizer, test_df, batch_size=2, few_shot_df=few_shot_df, isbenchmark=False)
    test_acc = (test_df["pred"] == test_df["ans"].map(LABEL_MAP)).mean()
    print(f"Test Accuracy: {test_acc:.4f}")
    tb_writer.add_scalar("test/accuracy", test_acc, final_step)
    tb_writer.flush()
    tb_writer.close()
    test_df.to_csv(f"test_split_results_{RUN_TS}.csv", index=False)
    print(
        f"Test split inference complete! Results saved to test_split_results_{RUN_TS}.csv"
    )

    # benchmark
    print("-" * 50)
    print("Running benchmark...")
    benchmark_df = pd.read_csv(BENCHMARK_PATH)
    predictions = run_inference(model, tokenizer, benchmark_df, batch_size=1, few_shot_df=few_shot_df, isbenchmark=True)
    letter_to_int = {"A": 0, "B": 1, "C": 2, "D": 3}
    submission = pd.DataFrame(
        {
            "question_id": benchmark_df.get("question_id", range(len(benchmark_df))),
            " pred": [f" {letter_to_int[p]}" for p in predictions],
        }
    )
    submission.to_csv(f"submission_{RUN_TS}.csv", index=False)
    print(f"Benchmark complete! Results saved to submission_{RUN_TS}.csv")


if __name__ == "__main__":
    main()
