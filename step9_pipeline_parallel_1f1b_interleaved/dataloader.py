import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from functools import partial
from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

import process_group_manager as pgm

class MicroBatchDataLoader(DataLoader):
    def __init__(self, seq_len, micro_batch_size, grad_acc_steps, dataset_name, tokenizer_name, max_tokens, num_workers, num_proc, seed, split="train"):
        
        self.micro_batch_size = micro_batch_size
        self.grad_acc_steps = grad_acc_steps
        self.seq_len = seq_len

        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)

        # Tokenize and chunk the dataset
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.seq_len, num_proc)
        
        total_tokens = self.tokenized_dataset.num_rows * (self.seq_len + 1)
        assert total_tokens >= max_tokens, f"Not enough tokens. Have {total_tokens} tokens but need {max_tokens} tokens"

        self.sampler = DistributedSampler(
            self.tokenized_dataset, 
            num_replicas=pgm.process_group_manager.dp_world_size, 
            rank=pgm.process_group_manager.dp_rank, 
            seed=seed,
            shuffle=False
        )

        super().__init__(
            self.tokenized_dataset,
            batch_size=micro_batch_size,
            collate_fn=self.collate_batch, 
            pin_memory=True, 
            num_workers=num_workers, 
            sampler=self.sampler,
            shuffle=False,
        )

    def tokenizer_group_text(self, examples, tokenizer, sequence_length):
        """Tokenize a list of texts and group them in chunks of sequence_length + 1"""
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='np'
        )
        concatenated_tokens = {'input_ids': np.concatenate(tokenized_text_batch['input_ids'])}
        total_length = len(concatenated_tokens['input_ids'])
    
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1

        result = {
            'input_ids': [
                concatenated_tokens['input_ids'][i : i + sequence_length + 1]
                for i in range(0, total_length - sequence_length, sequence_length)
            ]
        }
        return result

    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc):
        """Tokenize the dataset and group texts in chunks of sequence_length + 1"""
        tokenizer_func = partial(
            self.tokenizer_group_text,
            tokenizer=self.tokenizer,
            sequence_length=sequence_length
        )

        tokenized_dataset = dataset.map(
            tokenizer_func,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)
            }),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True, # Preprocess dataset only once and cache it
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )

        return tokenized_dataset

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        batch_size = batch_input_ids.size(0)
        input_ids = batch_input_ids[:, :-1].contiguous()
        target_ids = batch_input_ids[:, 1:].contiguous()
        position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
        attn_mask = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "hidden_states": None
        }
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise StopIteration
        return batch