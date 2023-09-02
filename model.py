import copy

import math

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import util
import logging
from typing import Iterable
import numpy as np
import torch.nn.init as init
import higher_order as ho
import wandb


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()



def create_positional_embeddings_random(smaller_embeddings, larger_embeddings):
    target_size = larger_embeddings.size()[0]
    return torch.cat((smaller_embeddings, torch.rand([target_size - smaller_embeddings.size()[0], smaller_embeddings.size()[1]])))

def create_positional_embeddings_repeated(smaller_embeddings, larger_embeddings):
    target_size = larger_embeddings.size()[0]
    ret = torch.zeros([target_size, smaller_embeddings.size()[1]])
    i = 0
    for index in range(0, target_size, smaller_embeddings.size()[0]):
        length = min(smaller_embeddings.size()[0], ret.size()[0] - index)
        ret[index: index + length, :] = smaller_embeddings[:length, :]
        i += 1
    return ret

def create_positional_embeddings_lin_proj(smaller_embeddings, larger_embeddings):
    mapping_embeddings = larger_embeddings[:smaller_embeddings.size()[0], :]
    transformation_matrix = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(mapping_embeddings.T, mapping_embeddings)), mapping_embeddings.T), smaller_embeddings)
    return torch.matmul(larger_embeddings, transformation_matrix)

def create_positional_embeddings_original(smaller_embeddings, larger_embeddings):
    return larger_embeddings

def average_models(original_weights, new_weights):
    ret = {}
    count = len(original_weights.keys())
    i = 0
    for name in original_weights.keys():
        if name not in new_weights:
            continue
        if original_weights[name].size() == new_weights[name].size():
            ret[name] = (i/count) * new_weights[name] + (1 - i/count) * original_weights[name]
        else:
            ret[name] = original_weights[name]
        i += 1
    ret["embeddings.position_ids"] = original_weights["embeddings.position_ids"]
    ret["embeddings.word_embeddings.weight"] = original_weights["embeddings.word_embeddings.weight"]
    ret["embeddings.position_embeddings.weight"] = original_weights["embeddings.position_embeddings.weight"]
    ret["embeddings.token_type_embeddings.weight"] = original_weights["embeddings.token_type_embeddings.weight"]
    ret["embeddings.LayerNorm.weight"] = original_weights["embeddings.LayerNorm.weight"]
    ret["embeddings.LayerNorm.bias"] = original_weights["embeddings.LayerNorm.bias"]

    return ret



positional_mapping_types = {"random": create_positional_embeddings_random, "repeated": create_positional_embeddings_repeated, "linproj": create_positional_embeddings_lin_proj, "original": create_positional_embeddings_original}


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None, instructions=None):
        super().__init__()
        self.config = config
        self.device = device

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        bert_config = AutoConfig.from_pretrained(config['bert_pretrained_name_or_path'])
        bert_config.hidden_dropout_prob = config['bert_dropout_rate']
        bert_config.attention_probs_dropout_prob = config['bert_dropout_rate']
        model = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'], from_tf=config["from_tf"], config=bert_config)
        bert_config.return_dict = False
        if "bert_weights_name" in config:
            bert_config.max_position_embeddings = config["max_segment_len"] + 2
            weights_config = AutoConfig.from_pretrained(config["bert_weights_name"])
            bert_config.vocab_size = weights_config.vocab_size
            bert_config.type_vocab_size = weights_config.type_vocab_size
            # weights_config.max_position_embeddings = config["max_segment_len"]
            weightsModel = AutoModel.from_pretrained(config['bert_weights_name'], from_tf=config["from_tf"], config=weights_config)
            state_dict = weightsModel.state_dict()
            model_state_dict = model.state_dict()
            if weights_config.max_position_embeddings < bert_config.max_position_embeddings:
                # state_dict["embeddings.position_embeddings.weight"] = torch.cat((state_dict["embeddings.position_embeddings.weight"], torch.rand([bert_config.max_position_embeddings - weights_config.max_position_embeddings, weights_config.hidden_size])))
                state_dict["embeddings.position_embeddings.weight"] = positional_mapping_types[config["positional_mapping_type"]](state_dict["embeddings.position_embeddings.weight"], model_state_dict["embeddings.position_embeddings.weight"])[:bert_config.max_position_embeddings,:]
                # state_dict["embeddings.position_embeddings.weight"] = create_positional_embeddings_lin_proj(state_dict["embeddings.position_embeddings.weight"], model_state_dict["embeddings.position_embeddings.weight"])
                state_dict["embeddings.position_ids"] = torch.tensor(range(0, bert_config.max_position_embeddings)).reshape([1, bert_config.max_position_embeddings])
            self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'], from_tf=config["from_tf"], config=bert_config, state_dict=state_dict)
        else:
            self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'], from_tf=config["from_tf"], config=bert_config)


        if "combine_with" in config:
            new_config = AutoConfig.from_pretrained(config['combine_with'])
            new_config.return_dict = False
            new_model = AutoModel.from_pretrained(config['combine_with'], from_tf=config["from_tf"], config=new_config)
            new_state_dict = new_model.state_dict()
            old_state_dict = self.bert.state_dict()
            new_state_dict = average_models(old_state_dict, new_state_dict)
            self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'], from_tf=config["from_tf"], config=bert_config, state_dict=new_state_dict)

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        if config['use_trees']:
            self.trees_output_size = self.bert_emb_size
            # self.span_emb_size += self.trees_output_size

        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config['use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None
        self.emb_deprels = self.make_embedding(len(config['deprels'])) if config['use_trees'] else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
        if config['separate_singletons']:
            self.singletons_span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
            self.singletons_coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] else None
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'], [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config['use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] else None

        if config["model_singletons"]:
            self.singleton_embedding = self.make_embedding(1, emb_size=self.span_emb_size)
        self.trees_ffnn = self.make_ffnn((self.bert_emb_size + config["feature_emb_size"]) * config['tree_path_length'], config["tree_ffnn_size"], output_size=self.trees_output_size) if config['use_trees'] else None
        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config['coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config['higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'], [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if config['higher_order'] == 'cluster_merging' else None
        if config["span2head"]:
            if config["span2head_binary"]:
                self.span2head_ffnn = self.make_ffnn(self.span_emb_size + self.bert_emb_size, config["ffnn_size"], output_size=1)
            else:
                self.span2head_ffnn = self.make_ffnn(self.span_emb_size, config["ffnn_size"], output_size=self.max_span_width)
        if config["use_push_pop_detection"] and instructions is not None:
            self.push_pop_ffnn = self.make_ffnn(self.bert_emb_size, config["ffnn_size"], output_size=len(instructions))
        self.instructions = instructions
        self.update_steps = 0  # Internal use for debug
        self.debug = True

    def make_embedding(self, dict_size, std=0.02, emb_size=None):
        if emb_size == None:
            emb_size = self.config['feature_emb_size']
        emb = nn.Embedding(dict_size, emb_size)
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def extract_spans_from_push_pop(self, instructions):
        span_starts, span_ends = [], []
        instructions = [self.instructions[instruction - 1].split(",") if instruction > 0 else [] for instruction in instructions]
        stack = []
        for i, ins in enumerate(instructions):
            for instruction in ins[1:]:
                if instruction == "PUSH":
                    stack.append(len(span_ends))
                    span_starts.append(i)
                    span_ends.append(-1)
                else:
                    stack_index = int(instruction.split(":")[-1])
                    if stack_index > len(stack):
                        logger.warning("Invalid stack instruction: POP before PUSH.")
                        continue
                    span_ends[stack[-stack_index]] = i
                    stack.pop(-stack_index)
        return torch.tensor(span_starts, device=self.device), torch.tensor(span_ends, device=self.device)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                 is_training, parents, deprel_ids, instructions,
                                 heads=None, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

        # Get token emb
        mention_doc = self.bert(input_ids, attention_mask=input_mask)[0]  # [num seg, num max tokens, emb size]
        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]
        instructions = instructions[input_mask]
        num_words = mention_doc.shape[0]

        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        if conf["heads_only"]:
            candidate_starts = candidate_ends = torch.arange(0, num_words, device=device)
        else:
            candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
            candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            if conf["heads_only"]:
                same_span = (torch.unsqueeze(heads, 1) == torch.unsqueeze(candidate_starts, 0)).to(torch.long)
            else:
                same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
                same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
                same_span = (same_start & same_end).to(torch.long)
            #TODO add gold labels for singletons - probably do not
            if conf["model_singletons"]:
                same_cluster = torch.unsqueeze(gold_mention_cluster_map, 0) == torch.unsqueeze(gold_mention_cluster_map, 1)
                self.singletons = gold_mention_cluster_map[torch.squeeze(torch.sum(same_cluster, 1)) == 1]
                # mentions = gold_mention_cluster_map.detach().numpy()
                # mentions_counts = {}
                # for mention in mentions:
                #     if mention not in mentions_counts:
                #         mentions_counts[mention] = 1
                #     else:
                #         mentions_counts[mention] += 1
                # self.singletons = torch.tensor([k for k,v in mentions_counts.items() if v == 1]).to(self.device)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float), same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long), 0)  # [num candidates]; non-gold span has label 0
            if conf["span2head"]:
                candidate_span_heads = torch.matmul(torch.unsqueeze(heads + 1, 0).to(torch.float), same_span.to(torch.float))
                candidate_span_heads = torch.squeeze(candidate_span_heads.to(torch.long), 0)

        if conf['use_trees']:
            deprel_ids = deprel_ids[input_mask]

            deprels_emb = self.emb_deprels(deprel_ids.to(torch.long))
            deprels_emb_ext = torch.cat((deprels_emb, torch.zeros([1, deprels_emb.size(1)]).to(device)))

            parents = torch.transpose(parents, 1, 2)[input_mask]
            parents[(parents < 0) | (parents > num_words)] = num_words
            mention_doc_ext = torch.cat((mention_doc, torch.zeros([1, mention_doc.size(1)]).to(device)))
            # parents[parents == -1] = mention_doc.size(0)

            deprel_path = deprels_emb_ext[parents]
            token_path = mention_doc_ext[parents]

            deprel_path = torch.cat((torch.unsqueeze(deprels_emb, 1), deprel_path[:,0:-1,:]), dim=1)

            path_emb = torch.cat((deprel_path, token_path), dim=-1)

            tree_repr = self.trees_ffnn(path_emb.view(num_words, -1))

            mention_doc = mention_doc + tree_repr


        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu, candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        if self.config["use_push_pop_detection"]:
            instructions_logits = self.push_pop_ffnn(mention_doc)
            pp_span_starts, pp_span_ends = self.extract_spans_from_push_pop(torch.argmax(instructions_logits.cpu(), dim=-1))
            equal_starts = torch.unsqueeze(candidate_starts, dim=0) == torch.unsqueeze(pp_span_starts, dim=1)
            equal_ends = torch.unsqueeze(candidate_ends, dim=0) == torch.unsqueeze(pp_span_ends, dim=1)
            selected_idx_pp = torch.any((equal_starts & equal_ends), dim=0)
            selected_idx_pp[selected_idx] = True
            selected_idx = selected_idx_pp
            num_top_spans = torch.sum(selected_idx)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        if conf["span2head"]:
            if conf["span2head_binary"]:
                head_absolute_position = torch.unsqueeze(top_span_starts, 1).repeat(1, self.max_span_width) + torch.unsqueeze(torch.arange(self.max_span_width), 0).to(self.device)
                valid_head_mask = head_absolute_position <= torch.unsqueeze(top_span_ends, -1)
                span2head_embedding = torch.unsqueeze(top_span_emb, 1).repeat(1, self.max_span_width, 1)
                head_embedding = torch.zeros(num_top_spans, self.max_span_width, self.bert_emb_size).to(self.device)
                head_embedding[valid_head_mask] = mention_doc[head_absolute_position[valid_head_mask]]
                span2head_embedding = torch.cat([span2head_embedding, head_embedding], -1)
            else:
                span2head_embedding = top_span_emb
            span2head_logits = torch.squeeze(self.span2head_ffnn(span2head_embedding))
            if do_loss:
                top_span_heads = candidate_span_heads[selected_idx]
        else:
            span2head_logits = None
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        if conf["model_singletons"]:
            top_span_emb = torch.cat([top_span_emb, torch.unsqueeze(self.singleton_embedding(torch.tensor(0, device=device)), 0)], 0)
            top_span_mention_scores = torch.cat([top_span_mention_scores, torch.tensor([0], device=device)], 0)
            if top_span_cluster_ids is not None:
                top_span_cluster_ids = torch.cat([top_span_cluster_ids, torch.tensor([-100], device=device)], 0)
        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        if conf["model_singletons"]:
            num_top_spans += 1
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1) #Mask to triangular matrix
        if conf["model_singletons"]:
            antecedent_mask[:, -1] = True
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0)
        # TODO no nekam sem pridat embedding pro virtualni entecedent
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        if conf["model_singletons"] and conf["mask_singleton_binary_score"]:
            singleton_mask = torch.ones_like(pairwise_coref_scores)
            singleton_mask[-1, :] = 0
            singleton_mask[:, -1] = 0
            pairwise_coref_scores = pairwise_coref_scores * singleton_mask
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        if conf["model_singletons"]:
            num_top_spans -= 1
            antecedent_mask = antecedent_mask[:-1,:]
            pairwise_fast_scores = pairwise_fast_scores[:-1,:]
            antecedent_offsets = antecedent_offsets[:-1,:]
            if conf["separate_singletons"]:
                pairwise_fast_scores[:, -1] += torch.squeeze(self.singletons_span_emb_score_ffnn(candidate_span_emb[selected_idx, :]), 1)


        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx, device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                if conf["model_singletons"]:
                    top_span_starts = torch.cat([top_span_starts, torch.tensor([0], device=device)], 0)
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                if conf["model_singletons"]:
                    top_antecedent_speaker_id = torch.cat([top_antecedent_speaker_id, -1 * torch.ones([1, max_top_antecedents], device=device)], 0)
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                if conf["model_singletons"]:
                    same_speaker = same_speaker[:-1,:]
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents, 1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                if conf["model_singletons"]:
                    top_antecedent_seg_ids = torch.cat([top_antecedent_seg_ids, torch.zeros([1, max_top_antecedents], dtype=torch.long, device=device)], 0)
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0, self.config['max_training_sentences'] - 1)
                if conf["model_singletons"]:
                    top_antecedent_seg_distance = top_antecedent_seg_distance[:-1,:]
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                feature_emb = torch.cat(feature_list, dim=2)
                if conf["model_singletons"]:
                    top_span_emb = top_span_emb[:-1, :]
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                if conf['separate_singletons']:
                    top_pairwise_slow_scores[top_antecedent_idx == top_pairwise_slow_scores.shape[0]] += torch.squeeze(self.singletons_coref_score_ffnn(pair_emb), 2)[top_antecedent_idx == top_pairwise_slow_scores.shape[0]]
                if conf["model_singletons"] and conf["mask_singleton_binary_score"]:
                    top_pairwise_slow_scores[top_antecedent_idx == top_pairwise_slow_scores.shape[0]] = 0
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores, self.emb_cluster_size, self.cluster_score_ffnn, None, self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'], easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores, self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
            if conf["model_singletons"]:
                top_span_starts = top_span_starts[:-1]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)  # [num top spans, max top antecedents + 1]
            return candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores, span2head_logits


        # Get gold labels

        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        if conf["model_singletons"]:
            if conf["subtract_best_from_singleton"]:
                non_singleton_scores = torch.clone(top_pairwise_scores)
                non_singleton_scores[top_antecedent_idx == top_pairwise_scores.shape[0]] = 0
                top_pairwise_scores[top_antecedent_idx == top_pairwise_scores.shape[0]] -= torch.maximum(torch.max(non_singleton_scores, 1)[0][torch.max(top_antecedent_idx, 1)[0] == top_pairwise_scores.shape[0]], torch.tensor(0))
            if top_span_cluster_ids is not None:
                top_span_cluster_ids = top_span_cluster_ids[:-1]
            singletons_mask = torch.squeeze((torch.unsqueeze(top_span_cluster_ids, 0) == torch.unsqueeze(self.singletons, 1)).any(0))
            # sigleton_gold_labels = (top_antecedent_cluster_ids == -100) & torch.unsqueeze(singletons_mask[top_span_cluster_ids >= 0], 1)
            sigleton_gold_labels = (top_antecedent_cluster_ids == -100)
            if not conf["model_mentions"]:
                sigleton_gold_labels &= torch.unsqueeze(singletons_mask, 1)
            else:
                sigleton_gold_labels &= torch.unsqueeze(top_span_cluster_ids > 0, 1)
        top_antecedent_cluster_ids += (top_antecedent_mask.to(torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        if conf["model_singletons"]:
            pairwise_labels = pairwise_labels | sigleton_gold_labels
            # pairwise_labels = top_antecedent_cluster_ids == -100
            if top_span_cluster_ids is not None:
                top_span_cluster_ids = torch.cat([top_span_cluster_ids, torch.tensor([-100], device=device)], 0)
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1) # pridava score pro dummy
        if conf['loss_type'] == 'marginalized':
            if conf["model_mentions"]:
                mention_scores = top_pairwise_scores[top_antecedent_idx == top_pairwise_scores.shape[0]]
                mention_scores = torch.stack((mention_scores, -mention_scores), dim=1)
                mention_gold_scores = pairwise_labels.to(torch.float)[top_antecedent_idx == top_pairwise_scores.shape[0]]
                mention_gold_scores = torch.stack((torch.log(mention_gold_scores), torch.log(1 - mention_gold_scores)), dim=1)
                top_pairwise_scores[top_antecedent_idx == top_pairwise_scores.shape[0]] = -math.inf
                pairwise_labels[top_antecedent_idx == top_pairwise_scores.shape[0]] = False
                dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
                top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)
                top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1) # pridava score pro dummy
                log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
                log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
                loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
                loss += torch.sum(torch.logsumexp(mention_scores, dim=1) - torch.logsumexp(mention_scores + mention_gold_scores, dim=1))
            else:
                log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
                log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
                loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif conf['loss_type'] == 'hinge':
            top_antecedent_mask = torch.cat([torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * conf['mention_loss_coef']
            loss += loss_mention

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm
        span2head_loss = 0.0
        gold_heads = 0
        gold_heads2 = 0
        if conf["span2head"]:
            top_span_head_offsets = top_span_heads - top_span_starts - 1
            top_span_head_offsets[top_span_head_offsets < 0] = -100
            loss_fn = nn.BCELoss()
            if torch.any(top_span_head_offsets >= 0):
                span2head_loss = 1000 * loss_fn(torch.sigmoid(span2head_logits[top_span_head_offsets >= 0, :]), one_hot(top_span_head_offsets[top_span_head_offsets >= 0], span2head_logits.size(dim=1), self.device))
                loss += span2head_loss
                gold_heads = (top_span_heads > 0).sum()
                gold_heads2 = (top_span_head_offsets >= 0).sum()
        if conf["use_push_pop_detection"]:
            pp_loss_fn = torch.nn.CrossEntropyLoss()
            pp_loss = pp_loss_fn(torch.transpose(instructions_logits, -1, 1), instructions)
            loss += 100 * pp_loss
        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum()/num_top_spans))
                logger.info('Number of gold spans: %d, mention identification recall: %.2f' % (gold_starts.shape[0], (top_span_cluster_ids > 0).sum() / gold_starts.shape[0]))
                metrics = {"spans_ratio": (top_span_cluster_ids > 0).sum()/num_top_spans, "mention_recall": (top_span_cluster_ids > 0).sum() / gold_starts.shape[0]}
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                    metrics["mention_loss"] = loss_mention
                if conf['loss_type'] == 'marginalized':
                    logger.info('norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                    metrics["loss"] = loss
                else:
                    logger.info('loss: %.4f' % loss)
                    metrics["loss"] = loss
                if conf["span2head"]:
                    logger.info("span2head loss: %.4f" % span2head_loss)
                    metrics["span2head_loss"] = span2head_loss
                    logger.info("Number of gold heads: %d, %d" % (gold_heads, gold_heads2))
                if conf["use_push_pop_detection"]:
                    logger.info(f"PUSH/POP loss: {pp_loss}")
                    metrics["PUSH_POP_loss"] = pp_loss
                wandb.log(metrics)
        self.update_steps += 1

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores, span2head_logits], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx, key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        antecedent_scores = np.array(antecedent_scores)
        # antecedent_scores[np.concatenate([np.zeros([np.shape(antecedent_scores)[0], 1], dtype=np.bool), np.array(antecedent_idx) == len(antecedent_idx)], axis=1)] = 1000000
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)
        if self.config["model_mentions"]:
            altered_scores = np.copy(antecedent_scores)
            altered_scores[torch.cat((-100 * torch.ones([len(antecedent_idx), 1]), torch.tensor(antecedent_idx)), 1) == len(antecedent_idx)] = -math.inf
            predicted_antecedents_without_mentions = self.get_predicted_antecedents(antecedent_idx, altered_scores)
        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            if predicted_idx == len(antecedent_idx):
                if not self.config["model_mentions"] or predicted_antecedents_without_mentions[i] < 0:
                    cluster_id = len(predicted_clusters)
                    mention = (int(span_starts[i]), int(span_ends[i]))
                    predicted_clusters.append([mention])
                    mention_to_cluster_id[mention] = cluster_id
                    continue
                else:
                    predicted_idx = predicted_antecedents_without_mentions[i]

            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def get_mention2head_map(self, span_starts, span_ends, heads, mention2head_map=None):
        if mention2head_map is None:
            mention2head_map = {}
        for start, end, head in zip(span_starts, span_ends, heads):
            mention2head_map[str(start) + "-" + str(end)] = head
        return mention2head_map

    def predict_heads(self, span_starts, span_ends, head_logits):
        valid_head_mask = torch.zeros_like(head_logits) + torch.unsqueeze(torch.arange(head_logits.size(-1)), 0).to(self.device)
        valid_head_mask = valid_head_mask <= torch.unsqueeze(span_ends - span_starts, -1)
        head_logits[torch.logical_not(valid_head_mask)] = -np.inf  # discard heads out of span
        heads_binary = sigmoid(head_logits) > .5
        # heads_binary = torch.zeros_like(head_logits).to(torch.bool)
        no_head = torch.logical_not(torch.any(heads_binary, dim=1))
        if torch.any(no_head):
            if self.config["span2head_fallback"] == "best":
                heads_binary[no_head, :] = one_hot(torch.argmax(head_logits[no_head, :], dim=1), head_logits.size()[-1], self.device).to(torch.bool)  # maximum where no head is predicted
            elif self.config["span2head_fallback"] == "first":
                heads_binary[no_head, 0] = 1  # first word where no head is predicted
            else:
                heads_binary += valid_head_mask * torch.unsqueeze(no_head, dim=1)  # all words where no head is predicted
        heads = [list(np.where(row)[0]) for row in heads_binary.cpu().numpy()]
        return heads


    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
        return self.update_evaluator_from_clusters(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator)

    def update_evaluator_from_clusters(self, predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator):
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def merge_clusters(self, first_clusters, first_m2c, second_clusters, second_m2c):
        res_clusters = copy.deepcopy(first_clusters)
        res_m2c = copy.deepcopy(first_m2c)
        for cluster in second_clusters:
            found = False
            for mention in cluster:
                if mention in first_m2c:
                    cluster_id = first_m2c[mention]
                    found = True
                    if cluster == first_clusters[cluster_id]:
                        break
                    res_clusters[cluster_id] = tuple(sorted(list(set(first_clusters[cluster_id]) | set(cluster))))
                    for mention in cluster:
                        if mention in res_m2c and res_m2c[mention] != cluster_id:
                            res_clusters[res_m2c[mention]] = tuple(sorted(list(set(res_clusters[res_m2c[mention]]) - {mention})))
                        res_m2c[mention] = cluster_id
                    break
            if not found:
                for mention in cluster:
                    res_m2c[mention] = len(res_clusters)
                res_clusters.append(cluster)
        res_clusters = [cluster for cluster in res_clusters if len(cluster) > 0]
        res_m2c = {}
        for i, cluster in enumerate(res_clusters):
            for mention in cluster:
                res_m2c[mention] = i
        return res_clusters, res_m2c

    def filter_overlapping(self, clusters, m2c, end):
        res_clusters = []
        res_m2c = {}
        for i, cluster in enumerate(clusters):
            found = False
            for mention in cluster:
                if mention[1] > end - self.config["max_segment_len"]:
                    found = True
                    break
            if found:
                for mention in cluster:
                    res_m2c[mention] = len(res_clusters)
                res_clusters.append(cluster)
        return res_clusters, res_m2c

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def one_hot(a, num_classes, device, ignore_index=-100):
    a[a == ignore_index] = -1
    e = torch.cat([torch.zeros(1, num_classes), torch.eye(num_classes)], dim=0).to(device)
    return e[torch.flatten(a + 1)]

