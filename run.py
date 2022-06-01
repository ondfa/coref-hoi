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
import tensorflow as tf
import os
from functools import cmp_to_key

tf.config.set_visible_devices([], 'GPU')

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

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config, language=self.config.language)
        wandb_api_key = load_wandb_api_key(WANDB_API_KEY_DIR)
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
        import wandb
        wandb.init(project="coref-multiling", entity="ondfa", config=self.config, reinit=True, name=config_name + "_" + self.name_suffix)

    def initialize_model(self, saved_suffix=None):
        model = CorefModel(self.config, self.device)
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

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

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
                example_gpu = [d.to(self.device) for d in example]
                torch.cuda.empty_cache()
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
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=True, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
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
        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=True, conll_path=self.config['conll_test_path'], tb_writer=tb_writer, save_predictions=True, phase="test")
        if best_model_path is not None:
            logger.info('**********Best model evaluation**********')
            self.load_model_checkpoint(model, best_model_path[best_model_path.rindex("model_") + 6: best_model_path.rindex(".bin")])
            self.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=self.config['conll_test_path'], save_predictions=True, phase="best_model_eval")
        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None, save_predictions=False, phase="eval"):
        if isinstance(conll_path, list):
            for path in self.config['conll_test_path']:
                self.evaluate(model, tensor_examples, stored_info, step, official, path, tb_writer, save_predictions, phase)
            return 0.0, None
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        max_sentences = self.config["max_training_sentences"] if "max_pred_sentences" not in self.config else self.config["max_pred_sentences"]
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:7]  # Strip out gold
            num_sentences = tensor_example[0].shape[0]
            if num_sentences <= max_sentences:
                batch_examples = [tensor_example]
            else:
                batch_examples = Tensorizer(self.config, local_files_only=True, load_tokenizer=False).split_example(*tensor_example)
            predicted_clusters = []
            mention_to_cluster_id = {}
            for j, example in enumerate(batch_examples):
                example_gpu = [d.to(self.device) for d in example]
                with torch.no_grad():
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
                    sentence_len = tensor_example[3]
                    offset = j * max_sentences
                    word_offset = sentence_len[:offset].sum()
                    span_starts = span_starts + word_offset
                    span_ends = span_ends + word_offset
                example_gpu = [e.detach().cpu() for e in example_gpu]
                span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
                tmp_predicted_clusters, tmp_mention_to_cluster_id, _ = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
                predicted_clusters.extend(tmp_predicted_clusters)
                mention_to_cluster_id = {**tmp_mention_to_cluster_id, **mention_to_cluster_id}
            predicted_clusters = model.update_evaluator_from_clusters(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator)
            if self.config["filter_singletons"]:
                predicted_clusters = util.discard_singletons(predicted_clusters)
            doc_to_prediction[doc_key] = predicted_clusters
        dataset = conll_path.split("/")[-1].split("-")[0]
        p, r, f = evaluator.get_prf()
        metrics = {phase + '_Avg_Precision': p * 100, phase + '_Avg_Recall': r * 100, phase + '_Avg_F1': f * 100}
        metrics[phase + "_" + dataset + "_Avg_Precision"] = p * 100
        metrics[phase + "_" + dataset + "_Avg_Recall"] = r * 100
        metrics[phase + "_" + dataset + "_Avg_F1"] = f * 100
        if official:
            udapi_docs = udapi_io.map_to_udapi(udapi_io.read_data(conll_path), doc_to_prediction, stored_info['subtoken_maps'])

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
            fd.close()

        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            wandb.run.summary[name] = score
            if tb_writer:
                tb_writer.add_scalar(name, score, step)
        wandb.log(metrics)


        return f * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        for i, tensor_example in enumerate(tensor_examples):
            tensor_example = tensor_example[:7]
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
        import glob
        models = glob.glob(join(dir, '*model_*'))
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
    model = runner.initialize_model()

    runner.train(model)
