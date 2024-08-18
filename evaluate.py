from collections import defaultdict

import torch

from model import CorefModel
from run import Runner
import sys
from os.path import join


def eval(runner, model, stored_info, examples_dev, examples_test):
    runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'], save_predictions=True, phase="dev")  # Eval dev
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'], save_predictions=True, phase="test")  # Eval test


def evaluate(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()

    instructions = None
    deprels = None
    if runner.config["use_push_pop_detection"]:
        instructions = sorted(runner.data.stored_info["instructions_dict"].items(), key=lambda entry: entry[1])[1:]
        instructions = [ins[0] for ins in instructions]
    if runner.config["use_trees"]:
        deprels = sorted(runner.data.stored_info["deprels_dict"].items(), key=lambda entry: entry[1])
        deprels = [rel[0] for rel in deprels]
    # find_cross_example_coreference(examples_dev)
    if saved_suffix == "last":
        if "load_model_from_exp" in runner.config and runner.config["eval_parent"]:
            exp = runner.config["load_model_from_exp"]
        else:
            exp = runner.name
        model = CorefModel(runner.config, runner.device, instructions=instructions, subtoken_map=runner.data.stored_info["subtoken_maps"], deprels=deprels, bert_device=runner.bert_device, dtype=runner.float_dtype)
        runner.load_model_from_experiment(model, exp)
    else:
        model = runner.initialize_model(saved_suffix, instructions=instructions, subtoken_map=runner.data.stored_info["subtoken_maps"], deprels=deprels, bert_device=runner.bert_device, dtype=runner.float_dtype)

    stored_info = runner.data.get_stored_info()
    if runner.config["use_half_precision"]:
        model.to(device=runner.device, dtype=torch.bfloat16)
        model.bert.to(device=model.bert_device, dtype=torch.bfloat16)
    else:
        model.to(runner.device)
        model.bert.to(model.bert_device)
    if runner.config["use_half_precision"]:
        from torch.cuda.amp import autocast
        with autocast(dtype=torch.bfloat16):
             eval(runner, model, stored_info, examples_dev, examples_test)
    else:
         eval(runner, model, stored_info, examples_dev, examples_test)



if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    evaluate(config_name, gpu_id, 'last')
