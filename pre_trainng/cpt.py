from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

class LLMContinuedPreTrainer:
    def __init__(self, model_name: str, dataset_name: str, output_dir: str, training_args: TrainingArguments = None):
        """
        Initialize the pre-trainer with a model and dataset from Hugging Face.

        Args:
            model_name (str): Hugging Face model name (e.g., 'allenai/OLMo-2-1124-7B').
            dataset_name (str): Hugging Face dataset name (e.g., 'wikipedia').
            output_dir (str): Directory to save the trained model and checkpoints.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load dataset
        self.dataset = load_dataset(dataset_name)

        # Tokenize dataset
        self.tokenized_dataset = self.dataset.map(self._tokenize_function, batched=True, remove_columns=self.dataset["train"].column_names)

        # Set training arguments with defaults specific to the model
        self.training_args = training_args or self._get_default_training_args()

    def _tokenize_function(self, examples):
        """Tokenize dataset examples."""
        return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    def _get_default_training_args(self):
        """Fetch model-specific training arguments based on its recommended configuration."""
        if 'OLMo' in self.model_name:
            batch_size = 16  # Example default for OLMo; adjust as needed
            learning_rate = 2e-5
            num_train_epochs = 3
        elif 'LLaMA' in self.model_name:
            batch_size = 32  # Example default for LLaMA; adjust as needed
            learning_rate = 1e-4
            num_train_epochs = 3
        else:
            raise ValueError(f"Model {self.model_name} not recognized for default training args.")

        return TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=2,
            fp16=True,
            push_to_hub=False,
        )

    def train(self):
        """Start training with the Trainer API."""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'] if 'validation' in self.tokenized_dataset else None,
        )

        trainer.train()

        # Save the final model
        trainer.save_model(self.output_dir)
        print(f"Model saved to {self.output_dir}")

# Example Usage
if __name__ == "__main__":
    model_name = "allenai/OLMo-2-1124-7B"
    dataset_name = "wikipedia"
    output_dir = "./olmo-continued-pretraining"

    trainer = LLMContinuedPreTrainer(model_name, dataset_name, output_dir)
    trainer.train()
