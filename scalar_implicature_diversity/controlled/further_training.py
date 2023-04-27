'''
This code snippet is used to do further pre-training using SI-100 and SI-500
'''

from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse
import torch


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for training dataset')
    parser.add_argument('--output_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for output models')
    parser.add_argument('--model_name',
                        default=None,
                        type=str,
                        required=True,
                        help='BERT/RoBERTa')
    parser.add_argument('--device',
                        default=None,
                        type=str,
                        required=True,
                        help='Device for processing')
    args = parser.parse_args()

    if args.model_name == 'BERT':
        config = BertConfig.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        model = BertForMaskedLM.from_pretrained("bert-large-uncased", config=config)
    elif args.model_name == 'RoBERTa':
        config = RobertaConfig.from_pretrained('roberta-large')
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = RobertaForMaskedLM.from_pretrained("roberta-large", config=config)
    else:
        raise Exception('Unknown model. Try again')

    device = torch.device(args.device)
    model.to(device)

    # Load dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.dataset_path,
        block_size=256,
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.out_path,
        save_strategy="epoch",
        logging_strategy="epoch",
        prediction_loss_only=True,
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        fp16=True,
        num_train_epochs=10,
        logging_steps=500,
        report_to='wandb',
        gradient_accumulation_steps=4,
    )

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print('Training completed.')


if __name__ == '__main__':
    main()
