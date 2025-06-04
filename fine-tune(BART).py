import nltk
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration# pre-trained model
TrainingArguments

tokenizer = BartTokenizer.from_pretrained ("facebook/bart-large-cnn") #BART model
dataset = load_dataset ("cnn_dailymail", "3.0.0") #CNN/Daily Mail for fine-tuning the model

def preprocess(examples):
    inputs = [article for article in examples["article"]]
    #truncation- limiting the number of digits right of the decimal point
    model_inputs = tokenizer(examples["highlights"], max_length=128, truncation=True, padding = "max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding = "max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns = ["article", "highlights", "id"])
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

training_args = TrainingArguments(
    output_dir = "./bart_summarizer",
    evaluation_stratergy = "epoch",
    learnng_rate = 3e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    weight_decay = 0.01,
    save_total_limit = 2,
    num_train_epochs = 3,
    predicts_with_generate = True,
    logging_dir = "./logs",
    logging_steps = 100,
    )

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets ["train"].select(range(20000)),
    eval_dataset = tokenized_datasets ["valdation"].select(range(2000)),
    tokenizer = tokenizer,
    )

trainer.train()
model.save_pretrained("./fine-tune(BART)")
tokenizer.save_pretrained("./fine-tune(BART)")

print("model fine-tuned and saved to ./fine-tune(BART)")
    

    
