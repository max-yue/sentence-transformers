from torch import Tensor
from torch import nn
from transformers import AlbertModel, BertTokenizer
import json
from typing import Union, Tuple, List, Dict, Optional
import os
import numpy as np
import logging

class ALBERT(nn.Module):
    """ALBERT model to generate token embeddings.

    Each token is mapped to an output vector from BERT.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: Optional[bool] = None, model_args: Dict = {}, tokenizer_args: Dict = {}):
        super(ALBERT, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if max_seq_length > 510:
            logging.warning("BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length

        if self.do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        self.albert = AlbertModel.from_pretrained(model_name_or_path, **model_args)
        # self.tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)



    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.albert(**features)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.albert.config.output_hidden_states:
            hidden_states = output_states[2]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.albert.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3 #Add space for special tokens
        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt')


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.albert.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_albert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_albert_config.json')) as fIn:
            config = json.load(fIn)
        return ALBERT(model_name_or_path=input_path, **config)


if __name__ == "__name__":
    from transformers import *
    import torch
    from torch.nn.functional import softmax

    pretrained = 'voidful/albert_chinese_tiny'
    # tokenizer = BertTokenizer.from_pretrained(pretrained)
    # model = AlbertForMaskedLM.from_pretrained(pretrained)
    model = ALBERT(pretrained)

    inputtext = "今天[MASK]情很好"

    maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

    input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, masked_lm_labels=input_ids)
    loss, prediction_scores = outputs[:2]
    logit_prob = softmax(prediction_scores[0, maskpos]).data.tolist()
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, logit_prob[predicted_index])



