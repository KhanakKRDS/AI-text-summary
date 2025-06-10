import nltk
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration# pre-trained model
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq

dataset = load_dataset ("d0rj/wikisum") #wiki_sum for fine-tuning the model
subset = dataset["train"].select(range(10000)) #select 10000 examples instead of full dataset(reduce memory usage)
train_idx, val_idx = train_test_split(range(len(subset)), test_size = 0.2, random_state = 42)
subset_train = subset.select(train_idx)
subset_val = subset.select(val_idx)
tokenizer = BartTokenizer.from_pretrained ("facebook/bart-large-cnn") #BART tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn") #BART model

#tokenization (dynamically adjusts padding and trunction)
def preprocess(examples):
    #truncation - limiting the number of digits right of the decimal point
    model_inputs = tokenizer(examples["article"], max_length = 1024, truncation=True, padding = "longest")
    labels = tokenizer(examples["summary"], max_length = 256, truncation=True, padding = "longest")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#tokeniztion while removing large articles for efficiency    
tokenized_train = subset_train.map(preprocess, batched=True, remove_columns = ["article", "summary"])
tokenized_val = subset_val.map(preprocess, batched=True, remove_columns = ["article", "summary"])
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#define training prameters with optimizations
training_args = TrainingArguments(
    output_dir = "./checkpoints", #saves model checkpoints and final model (place where this script is saved)
    evaluation_strategy = "epoch", #evaluate after each epoch
    save_strategy = "epoch",
    learning_rate = 3e-5, #fine-tuning learning rates
    per_device_train_batch_size = 4,#small batch size to prevent memory overload
    gradient_accumulation_steps = 8, #accumulate gradients for larger effective batches
    per_device_eval_batch_size = 4, #evaluation batch size
    weight_decay = 0.01, #regularization to prevent overfitting
    save_total_limit = 2, #limits number of saved checkpoints
    num_train_epochs = 3, #number of training epochs
    #predict_with_generate = True,
    logging_dir = "./logs", #saves logs for monitoring training(via TensorBoard)
    logging_steps = 50, #log training progress every 50 steps
    report_to = ["tensorboard"], #enables tensorboard for tracking performances
    metric_for_best_model = "eval_loss",
    load_best_model_at_end = True,
    greater_is_better = False, #lower eval_loss is better
    predict_with_generate = True,
    )

#train model on selected dataset(define trainer with early stopping)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,  #using 20k samples for training
    eval_dataset = tokenized_val, #using 2k samples for validation
    data_collator = data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)], #stops training if validation loss doesn't improve
    )

#train the model with the improved settings
trainer.train() #train the model
#save the fine-tuned model
model.save_pretrained("My-tune-model")
tokenizer.save_pretrained("My-tune-model")

print("model fine-tuned and saved to My-tune-model)")
    
