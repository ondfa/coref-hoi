from model import CorefModel
from run import Runner
import sys
from os.path import join


def evaluate(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    if saved_suffix == "last":
        if "load_model_from_exp" in runner.config and runner.config["load_model_from_exp"]:
            exp = runner.config["load_model_from_exp"]
        else:
            exp = runner.name
        model = CorefModel(runner.config, runner.device)
        runner.load_model_from_experiment(model, exp)
    else:
        model = runner.initialize_model(saved_suffix)

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'], save_predictions=True, phase="dev")  # Eval dev
    # print('=================================')
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'], save_predictions=True, phase="test")  # Eval test
    # runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'], save_predictions=True, phase="dev")  # Eval test



if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    evaluate(config_name, gpu_id, 'last')
