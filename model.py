import warnings

from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertModel, RobertaForMaskedLM
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import _collate_batch
from transformers.modeling_bert import BertEmbeddings
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPooling
from transformers.modeling_roberta import RobertaEmbeddings
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

# 修改Bert的embedding层
# 第一个调用这个类，获取这个类的output

import torch.nn as nn
import torch
from config import CONFIG

maxlen_t = CONFIG['maxlen_text_concat']
code_prompt_lenth = CONFIG['code_prompt_lenth']

class Model(nn.Module):
    def __init__(self, encoder1, encoder2):
        super(Model, self).__init__()
        # encoder就是roberta模型，RobertaForMaskedLM
        self.bertEncoder = encoder1
        self.codeEncoder = encoder2

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, labels_code=None, num=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            #inputs_embeddings 是[batch_size,input_length,embedding_size]--[16,320,768]
            # inputs_embeddings = self.codeEncoder.embeddings.word_embeddings(code_inputs)
            # nl_embeddings = self.codeEncoder.roberta.embeddings.word_embeddings(nl_inputs)
            inputs_embeddings = self.codeEncoder.roberta.embeddings.word_embeddings(code_inputs)

            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings*(~nodes_mask)[:, :, None]+avg_embeddings*nodes_mask[:, :, None]

            return self.codeEncoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask,position_ids=position_idx)[0]
        else:
            return self.bertEncoder(code_inputs, attention_mask=code_inputs.ne(1))[1]





class BertForMaskedLM1(BertForMaskedLM):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super(BertForMaskedLM1, self).__init__(config)

        #         if config.is_decoder:
        #             logger.warning(
        #                 "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        #                 "bi-directional self-attention."
        #             )
        #         #用修改之后的子类
        self.bert = BertModel1(config, add_pooling_layer=False)

    #         self.cls = BertOnlyMLMHead(config)

    #         self.init_weights()

    #     def get_output_embeddings(self):
    #         return self.cls.predictions.decoder

    #     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #     @add_code_sample_docstrings(
    #         tokenizer_class=_TOKENIZER_FOR_DOC,
    #         checkpoint="bert-base-uncased",
    #         output_type=MaskedLMOutput,
    #         config_class=_CONFIG_FOR_DOC,
    #     )
    def forward(
            self,
            code_identity=None,
            code=None,
            code_nl=None,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            code_identity=code_identity,
            code=code,
            code_nl=code_nl,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertModel1(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        super(BertModel1, self).__init__(config)
        self.config = config
        # 用修改后的子类，本质是在embedding层修改
        self.embeddings = BertEmbeddings1(config)

    def forward(
            self,
            code_identity=None,
            code=None,
            code_nl=None,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            code_identity, code, code_nl, input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertEmbeddings1(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings1, self).__init__(config)

    #         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    #         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    #         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    #         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
    #         # any TensorFlow checkpoint file
    #         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    #         self.dropout = nn.Dropout(config.hidden_dropout_prob)

    #         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
    #         self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, code_identity=None, code=None, code_nl=None, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)  # token embedding, 大小为[batch_size, length, 768]的tensor
            # 拼接text的编码和code的编码
            # 循环batchsize
            for i in range(inputs_embeds.shape[0]):
                #判断需不需要把代码段拼进去
                if code_identity[i] == 1:
                    # 用k记录code部分写到哪一个字了，一条数据一个k
                    k = 0
                    # 不把code_prompt那句话拼进去
                    # if 1 <= k <= code_prompt_lenth:
                    #     k = k + 1
                    # else:
                    # 循环length,从180-299放上code的编码,保留最后一个SEP符号
                    for j in range(inputs_embeds.shape[1] - 1):
                        # 在后边的地方拼接
                        if j > maxlen_t - 1:
                            inputs_embeds[i][j] = code[i][k]
                            # 往后读一个k
                            k = k + 1
                k = 0
                for j in range(inputs_embeds.shape[1] - 1):
                    # 在后边的地方拼接
                    if j > 300 - 1:
                        inputs_embeds[i][j] = code_nl[i][k]
                        # 往后读一个k
                        k = k + 1

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings  # 三个embedding相加之后的结果，
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@dataclass
# 修改模型输入数据的格式
class DataCollatorForPrompt:
    # 用于得到符合特定任务的数据格式
    """
    For the prefix prompt designed by ourselves, do data processing and convert the words at the specified position into labels.
    对于我们自己设计的前缀提示，进行数据处理并将指定位置的单词转换为标签。
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked lanintguage modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
            是否使用屏蔽语言建模。如果设置为 ：obj：“False”，则标签与忽略padding tokens的输入一样（通过将其设置为 -100）。否则，标签为 -100
            非屏蔽令牌和要为屏蔽令牌预测的值。
        start_index (:obj:`int`, `optional`, defaults to 2):
            The start position of the clip that needs to be labeled.
        end_index(:obj:`int`, `optional`, defaults to 6):
            The end position of the clip that needs to be labeled.
    """
    tokenizer: PreTrainedTokenizerBase
    # tokenizer1: PreTrainedTokenizerBase
    #定义一个code的tokenizer
    mlm: bool = True
    text_start_index: int = 5
    text_end_index: int = 8

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # examples 是一个字典的list，利用以下代码使list转换成tensor


        """
        examples: 一个batch_size的dataset
        examples[0]：examples的第一条数据，以此类推

        examples是一个字典，里面有经过tokenizer之后的所有列
            格式为：
            [
            (注释：examples[0])    {'attention_mask': [..], 'code': [..], .. , 'special_tokens_mask': [..]},
            (注释：examples[1])    {'attention_mask': [..], 'code': [..], .. , 'special_tokens_mask': [..]},
            ...
                ]
        """
        # 判断examples[0]的类型是否与...相符合
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # special_tokens_mask在这里被删掉
        special_tokens_mask = batch.pop("special_tokens_mask", None)


        if self.mlm:
            # 把BertForMaskedLM需要的labels取出来
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], start_index=self.text_start_index, end_index=self.text_end_index, special_tokens_mask=special_tokens_mask
            )
            # batch["nl_ids"], batch['code_ids'], batch['labels_code'] = self.mask_tokens1(
            #     batch["code_ids"], batch["nl_ids"], start_index=self.code_start_index, end_index=self.code_end_index
            # )
        else:
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
            self, inputs: torch.Tensor, start_index, end_index, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把masked_indices改成自己定义的方式，不采用伯努利分布，
        将固定位置的input_ids来做labels，其他位置均为-100,
        input打标签的位置改成[MASK]
        """
        # 先把input_ids复制给labels


        labels = inputs.clone()

        # 然后把probability_matrix里所有值用0填满，形状和label一样
        probability_matrix = torch.zeros(labels.shape)

        # prompt设计的开始和结束位置（对于单prompt）
        start_index = start_index
        end_index = end_index

        # 将probability_matrix转换成bool类型
        masked_indices = probability_matrix.bool()

        # 根据start_index和start_index修改指定位置为True
        masked_indices[:, start_index:end_index] = True

        # 其他位置都是-100，不参与计算loss
        labels[~masked_indices] = -100
        # input相应的位置变成[MASK]
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

    def mask_tokens1(
            self, inputs: torch.Tensor, nl_inputs: torch.Tensor, start_index, end_index, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把masked_indices改成自己定义的方式，不采用伯努利分布，
        将固定位置的input_ids来做labels，其他位置均为-100,
        input打标签的位置改成[MASK]
        """
    
        # nl_tensor = torch.zeros(inputs.shape[0], CONFIG['nl_length'])
        all_inputs = torch.cat((nl_inputs, inputs), dim=1)
        # 先把input_ids复制给labels
        labels = all_inputs.clone()
    
        # 然后把probability_matrix里所有值用0填满，形状和label一样
        probability_matrix = torch.zeros(labels.shape)
    
        # prompt设计的开始和结束位置（对于单prompt）
        start_index = start_index
        end_index = end_index
    
        # 将probability_matrix转换成bool类型
        masked_indices = probability_matrix.bool()
    
        # 根据start_index和start_index修改指定位置为True
        masked_indices[:, start_index:end_index] = True
    
        # 其他位置都是-100，不参与计算loss
        labels[~masked_indices] = -100
        # input相应的位置变成[MASK]
        nl_inputs[:, start_index:end_index] = self.tokenizer1.convert_tokens_to_ids(self.tokenizer1.mask_token)
    
    
        return nl_inputs, inputs, labels
