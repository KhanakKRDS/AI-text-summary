import nltk
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration# pre-trained model
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

dataset = load_dataset ("d0rj/wikisum") #wiki_sum for fine-tuning the model
subset = dataset["train"].select(range(10000)) #select 10000 examples instead of full dataset(reduce memory usage)
tokenizer = BartTokenizer.from_pretrained ("facebook/bart-large-cnn") #BART model

#tokenization (dynamically adjusts padding and trunction)
def preprocess(examples):
    #truncation - limiting the number of digits right of the decimal point
    model_inputs = tokenizer(examples["article"], truncation=True, padding = "longest")
    labels = tokenizer(examples["summary"], truncation=True, padding = "longest")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#tokeniztion while removing large articles for efficiency    
tokenized_train = subset_train.map(preprocess, batched=True, remove_columns = ["article", "summary"])
tokenized_val = subset_val.map(preprocess, batched=True, remove_columns = ["article", "summary"])
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

train_dataset = tokenized_train
eval_dataset = tokenized_val

#define training prameters with optimizations
training_args = TrainingArguments(
    output_dir = "./", #saves model checkpoints and final model (place where this script is saved)
    eval_strategy = "epoch", #evaluate after each epoch
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
    )

#train model on selected dataset(define trainer with early stopping)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets ["train"].select(range(20000)), #using 20k samples for training
    eval_dataset = tokenized_datasets ["validation"].select(range(2000)), #using 2k samples for validation
    tokenizer = tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)], #stops training if validation loss doesn't improve
    )

#train the model with the improved settings
trainer.train() #train the model
#save the fine-tuned model
model.save_pretrained("./fine-tune-BART.py")
tokenizer.save_pretrained("./fine-tune-BART.py")

print("model fine-tuned and saved to ./fine-tune-BART.py)")
    
