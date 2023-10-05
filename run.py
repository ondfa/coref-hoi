import os
# os.environ['WANDB_DISABLED'] = 'true'
import requests

os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

import argparse
import logging
import random
import subprocess
import tempfile

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam, SGD

import udapi_io
from tensorize import CorefDataProcessor, Tensorizer
import util
import time
from os.path import join
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import CorefModel
import conll
import sys
# import tensorflow as tf

import shutil
from functools import cmp_to_key

# tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

WANDB_API_KEY_DIR = 'wandb_private/wandbkey.txt'

def load_wandb_api_key(path):
    with open(path, "r", encoding='utf-8') as f:
        data = f.read().replace('\n', '')

    data = data.strip()
    return data

def evaluate_coreud(gold_path, pred_path):
    cmd = ["python", "corefud-scorer/corefud-scorer.py", gold_path, pred_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    # if stderr is not None:
    #     logger.error(stderr)
    logger.info("Official result for {}".format(pred_path))
    logger.info(stdout)
    import re
    result = re.search(r"CoNLL score: (\d+\.?\d*)", stdout)
    if result is None:
        score = 0.0
    else:
        score = float(result.group(1))

    cmd = ["python", "corefud-scorer/corefud-scorer.py", gold_path, pred_path, "-s"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    # if stderr is not None:
    #     logger.error(stderr)
    logger.info("Official result with singletons for {}".format(pred_path))
    logger.info(stdout)
    result = re.search(r"CoNLL score: (\d+\.?\d*)", stdout)
    if result is None:
        score_with_singletons = 0.0
    else:
        score_with_singletons = float(result.group(1))
    return score, score_with_singletons


class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("experiment_name")
        self.parser.add_argument("gpu_id")
        for key, value in self.config.items():
            if type(value) == bool:
                self.parser.add_argument("--" + key, default=value, action="store_true")
            else:
                self.parser.add_argument("--" + key, default=value, type=type(value))
        for key, value in vars(self.parser.parse_args()).items():
            if key in self.config:
                self.config[key] = value

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id < 0 else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config, language=self.config.language)
        wandb_api_key = load_wandb_api_key(WANDB_API_KEY_DIR)
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
        while True:
            try:
                wandb.init(project="coref-multiling", entity="zcu-nlp", config=self.config, reinit=True, name=config_name + "_" + self.name_suffix)
                break
            except (requests.exceptions.ConnectionError, ConnectionRefusedError, wandb.errors.UsageError) as e:
                logger.error(e)
                time.sleep(5)

    def initialize_model(self, saved_suffix=None, **kwargs):
        model = CorefModel(self.config, self.device, **kwargs)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        if "load_model_from_exp" in self.config:
            self.load_model_from_experiment(model, self.config["load_model_from_exp"])
        return model

    def train(self, model):
        best_model_path = None
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        if conf["add_dev_to_train"]:
            examples_train.extend(examples_dev)
        stored_info = self.data.get_stored_info()
        if model is None:
            instructions = None
            deprels = None
            if conf["use_push_pop_detection"]:
                instructions = sorted(stored_info["instructions_dict"].items(), key=lambda entry: entry[1])[1:]
                instructions = [ins[0] for ins in instructions]
            if conf["use_trees"]:
                deprels = sorted(runner.data.stored_info["deprels_dict"].items(), key=lambda entry: entry[1])
                deprels = [rel[0] for rel in deprels]
            model = self.initialize_model(instructions=instructions, subtoken_map=stored_info["subtoken_maps"], deprels=deprels)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))
        model.to(self.device)

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                model.subtoken_map = torch.tensor(stored_info["subtoken_maps"][doc_key]).to(self.device)
                example_gpu = [d.to(self.device) for d in example]
                torch.cuda.empty_cache
                _, loss = model(*example_gpu)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time
                        wandb.log({"train_loss": avg_loss, "lr_bert": schedulers[0].get_last_lr()[0], "lr_task": schedulers[1].get_last_lr()[-1], "epoch": epo})
                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))
                    example_gpu = [e.detach().cpu() for e in example_gpu]
                    # Evaluate
                    if (len(loss_history) > 0 or conf['evaluate_first']) and len(loss_history) % conf['eval_frequency'] == 0:
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=True, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                        torch.cuda.empty_cache()
                        if f1 > max_f1:
                            max_f1 = f1
                            new_path = self.save_model_checkpoint(model, len(loss_history))
                            if best_model_path is not None:
                                os.remove(best_model_path)
                            best_model_path = new_path
                        logger.info('Eval max f1: %.2f' % max_f1)
                        start_time = time.time()


        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))
        logger.info('**********Dev eval**********')
        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
        logger.info('**********Test eval**********')
        #TODO test data eval
        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=True, conll_path=self.config['conll_test_path'], tb_writer=tb_writer, save_predictions=True, phase="test")
        if best_model_path is not None:
            logger.info('**********Best model evaluation**********')
            self.load_model_checkpoint(model, best_model_path[best_model_path.rindex("model_") + 6: best_model_path.rindex(".bin")])
            logger.info('**********Dev eval**********')
            self.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=self.config['conll_eval_path'], save_predictions=True, phase="best_model_eval")
            logger.info('**********Test eval**********')
            self.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=self.config['conll_test_path'], save_predictions=True, phase="best_model_test")
        # Wrap up
        tb_writer.close()
        return loss_history

    def find_cross_segment_coreference(self, examples, segment_len=1):
        cross_segment_corefs = []
        cross_examples = []
        for example in examples:
            segment_lens = example[1][3]
            cluster_ids = example[1][-1]
            mention_span_starts = example[1][-3]
            cluster_segment_ids = torch.sum(torch.unsqueeze(mention_span_starts, dim=0) > torch.unsqueeze(segment_lens, dim=1), dim=0)
            same_cluster = torch.triu(torch.unsqueeze(cluster_ids, dim=0) == torch.unsqueeze(cluster_ids, dim=1), diagonal=1)
            # different_segment = torch.triu(torch.abs(torch.unsqueeze(cluster_segment_ids, dim=0) - torch.unsqueeze(cluster_segment_ids, dim=1)) >= segment_len, diagonal=1)
            different_segment = torch.triu((torch.unsqueeze(cluster_segment_ids, dim=0) // segment_len) != (torch.unsqueeze(cluster_segment_ids, dim=1) // segment_len), diagonal=1)
            cross_segment = same_cluster & different_segment
            cross_segment_corefs.append(cross_segment)
            if torch.any(cross_segment):
                cross_examples.append(example)
                print("Cross-segment coreference found.")
        return cross_examples, cross_segment_corefs

    def filter_gold_data(self, tensor_examples, gold_data):
        res_docs = []
        ids = set([example[0] for example in tensor_examples])
        for doc in gold_data:
            if doc.meta["docname"] in ids:
                res_docs.append(doc)
        return res_docs

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None, save_predictions=False, phase="eval"):
        if isinstance(conll_path, list):
            for path in conll_path:
                self.evaluate(model, tensor_examples, stored_info, step, official, path, tb_writer, save_predictions, phase)
            return 0.0, None
        dataset = conll_path.split("/")[-1].split("-")[0]
        if self.config["eval_cross_segment"] or self.config["filter_long_mentions"]:
            gold_data = udapi_io.read_data(conll_path)
            if self.config["eval_cross_segment"]:
                tensor_examples, _ = self.find_cross_segment_coreference(tensor_examples, self.config["max_training_sentences"])
                gold_data = self.filter_gold_data(tensor_examples, gold_data)
            elif self.config["filter_long_mentions"]:
                gold_data = udapi_io.filter_long_mentions(gold_data, self.config["max_mention_length"])
            gold_fd = tempfile.NamedTemporaryFile("w", delete=True, encoding="utf-8")
            udapi_io.write_data(gold_data, gold_fd)
            gold_fd.flush()
            conll_path = gold_fd.name
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}
        doc_span_to_head = {}

        model.eval()
        torch.cuda.empty_cache()
        max_sentences = self.config["max_training_sentences"] if "max_pred_sentences" not in self.config else self.config["max_pred_sentences"]
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            model.subtoken_map = torch.tensor(stored_info["subtoken_maps"][doc_key]).to(self.device)
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:-4]  # Strip out gold
            num_sentences = tensor_example[0].shape[0]
            if num_sentences <= max_sentences:
                batch_examples = [tensor_example]
            else:
                batch_examples = Tensorizer(self.config, local_files_only=True, load_tokenizer=False).split_example(*tensor_example, step=1 if self.config["max_segment_overlap"] else None)
            predicted_clusters = []
            mention_to_cluster_id = {}
            span_to_head = {}
            # all_span_starts = []
            # all_span_ends = []
            # all_antecedent_idx = []
            # all_antecedent_scores = []
            # antecedent_offset = 0
            for j, example in enumerate(batch_examples):
                example_gpu = [d.to(self.device) for d in example]
                with torch.no_grad():
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, span2head_logits = model(*example_gpu)
                    offset = j if self.config["max_segment_overlap"] else j * max_sentences
                    # num_antecedents_in_first_segment = torch.sum(span_starts < example_gpu[3][0])
                    # if self.config["max_segment_overlap"]:
                    #     if j > 0 and example[0].shape[0] > 1:
                            # original_ids = torch.arange(span_starts.shape[0], device=span_starts.device)[span_starts >= torch.sum(example_gpu[3]) - example_gpu[3][-1]]
                            # span_ends = span_ends[span_starts >= torch.sum(example_gpu[3]) - example_gpu[3][-1]]
                            # shifts = original_ids - torch.arange(span_ends.shape[0], device=span_starts.device)
                            # antecedent_idx = antecedent_idx[span_starts >= torch.sum(example_gpu[3]) - example_gpu[3][-1], :]
                            # antecedent_idx -= torch.unsqueeze(shifts, dim=1)
                            # antecedent_scores = antecedent_scores[span_starts >= torch.sum(example_gpu[3]) - example_gpu[3][-1], :]
                            # span_starts = span_starts[span_starts >= torch.sum(example_gpu[3]) - example_gpu[3][-1]]
                    # antecedent_idx += antecedent_offset
                    # if self.config["max_segment_overlap"]:
                    #     antecedent_offset += num_antecedents_in_first_segment
                    # else:
                    #     antecedent_offset += span_starts.shape[0]
                    sentence_len = tensor_example[3]
                    word_offset = sentence_len[:offset].sum()
                    span_starts = span_starts + word_offset
                    span_ends = span_ends + word_offset
                example_gpu = [e.detach().cpu() for e in example_gpu]
                if span2head_logits is not None:
                    heads = model.predict_heads(span_starts, span_ends, span2head_logits)
                    model.get_mention2head_map(span_starts.tolist(), span_ends.tolist(), heads, span_to_head)
                span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
                # all_span_starts.extend(span_starts)
                # all_span_ends.extend(span_ends)
                # all_antecedent_idx.extend(antecedent_idx)
                # all_antecedent_scores.extend(antecedent_scores)
                tmp_predicted_clusters, tmp_mention_to_cluster_id, _ = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
                if self.config["max_segment_overlap"] and self.config["filter_overlapping_mentions"] and j > 0:
                    tmp_predicted_clusters, tmp_mention_to_cluster_id = model.filter_overlapping(tmp_predicted_clusters, tmp_mention_to_cluster_id, sentence_len[:offset + max_sentences].sum() - 1)
                if self.config["max_segment_overlap"]:
                    predicted_clusters, mention_to_cluster_id = model.merge_clusters(predicted_clusters, mention_to_cluster_id, tmp_predicted_clusters, tmp_mention_to_cluster_id)
                else:
                    predicted_clusters.extend(tmp_predicted_clusters)
                    mention_to_cluster_id = {**tmp_mention_to_cluster_id, **mention_to_cluster_id}
            predicted_clusters = model.update_evaluator_from_clusters(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator)
            # predicted_clusters = model.update_evaluator(all_span_starts, all_span_ends, all_antecedent_idx, all_antecedent_scores, gold_clusters, evaluator)
            if self.config["filter_singletons"]:
                predicted_clusters = util.discard_singletons(predicted_clusters)
            doc_to_prediction[doc_key] = predicted_clusters
            if span2head_logits is not None:
                doc_span_to_head[doc_key] = span_to_head
        p, r, f = evaluator.get_prf()
        metrics = {phase + '_Avg_Precision': p * 100, phase + '_Avg_Recall': r * 100, phase + '_Avg_F1': f * 100}
        metrics[phase + "_" + dataset + "_Avg_Precision"] = p * 100
        metrics[phase + "_" + dataset + "_Avg_Recall"] = r * 100
        metrics[phase + "_" + dataset + "_Avg_F1"] = f * 100
        if official:
            udapi_docs = udapi_io.map_to_udapi(udapi_io.read_data(conll_path), doc_to_prediction, stored_info['subtoken_maps'], doc_span_to_head)
            if self.config["filter_long_mentions"]:
                udapi_docs = udapi_io.filter_long_mentions(udapi_docs, self.config["max_mention_length"])
            if save_predictions:
                path = join(self.config['log_dir'], self.name_suffix + "_pred_" + phase + "_" + conll_path.split("/")[-1])
                fd = open(path, "wt", encoding="utf-8")
            else:
                fd = tempfile.NamedTemporaryFile("w", delete=True, encoding="utf-8")
            udapi_io.write_data(udapi_docs, fd)
            fd.flush()
            score, score_with_singletons = evaluate_coreud(conll_path, fd.name)
            metrics[phase + "_corefud_score"] = score
            metrics[phase + "_corefud_score_with_singletons"] = score_with_singletons
            metrics[phase + "_" + dataset + "_corefud_score"] = score
            metrics[phase + "_" + dataset + "_corefud_score_with_singletons"] = score_with_singletons
            if save_predictions and self.config["final_path"]:
                if not os.path.exists(self.config["final_path"]):
                    os.makedirs(self.config["final_path"])
                shutil.copyfile(fd.name, os.path.join(self.config["final_path"], conll_path.split("/")[-1]))
            fd.close()
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            wandb.run.summary[name] = score
            if tb_writer:
                tb_writer.add_scalar(name, score, step)
        wandb.log(metrics)

        if self.config["eval_cross_segment"]:
            gold_fd.close()
        return f * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        for i, tensor_example in enumerate(tensor_examples):
            tensor_example = tensor_example[:9]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
            # SGD(model.get_params()[1], lr=self.config['task_learning_rate'], weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        # if step < 30000:
        #     return  # Debug
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)
        return path_ckpt

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


    def load_model_from_experiment(self, model, experiment_name):
        config = util.initialize_config(experiment_name)
        dir = config["log_dir"]
        if "model_suffix" in self.config:
            path_ckpt = join(config['log_dir'], f'model_{self.config["model_suffix"]}.bin')
            model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
            logger.info('Loaded model from %s' % path_ckpt)
        else:
            import glob
            models = glob.glob(join(dir, 'model_*'))
            models.sort(key=cmp_to_key(compare_models))
            model.load_state_dict(torch.load(models[-1], map_location=torch.device('cpu')), strict=False)

def compare_models(m1, m2):
    split1 = m1.split("/")[-1].split("_")
    split2 = m2.split("/")[-1].split("_")
    d1 = datetime.strptime(split1[1] + "_" + split1[2], '%b%d_%H-%M-%S')
    d2 = datetime.strptime(split2[1] + "_" + split2[2], '%b%d_%H-%M-%S')
    if d1 == d2:
        return 1 if split1[3] > split2[3] else -1
    else:
        return 1 if d1 > d2 else -1

if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = Runner(config_name, gpu_id)
    # model = runner.initialize_model()

    runner.train(model=None)
