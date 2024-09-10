import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from configs import base_config
from generate_data import create_dataloader


def evaluate(model, val_data, test_data, metrics_list, model_saver, writer, steps):
    with torch.no_grad():
        model.eval()
        val_scores = utils.evaluate(model, val_data, metrics_list)
        test_scores = utils.evaluate(model, test_data, metrics_list)
        print("Eval after {} steps:".format(steps))
        print("Val scores: ", val_scores)
        print("Test scores: ", test_scores)
        model.train()
    if writer is not None:
        for stat in val_scores:
            writer.add_scalar(f"{stat}/val", val_scores[stat], steps)
            writer.add_scalar(f"{stat}/test", test_scores[stat], steps)
    model_saver.visit(model, val_scores)


def train(config: base_config.Config, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Dnar(config).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    model_name = "{}_{}".format(config.algorithm, seed)
    model_saver = utils.ModelSaver(config.models_directory, model_name)

    train_data = create_dataloader(config, "train", seed=seed, device=device)
    val_data = create_dataloader(config, "val", seed=seed + 1, device=device)
    test_data = create_dataloader(config, "test", seed=seed + 2, device=device)

    if config.tensorboard_logs:
        writer = SummaryWriter(comment=f"-{model_name}")

    model.train()

    steps = 0
    while steps <= config.num_iterations:
        for batch in train_data:
            steps += 1

            _, loss = model(batch, writer, training_step=steps)
            assert not torch.isnan(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if steps % config.eval_each == 1:
                evaluate(
                    model,
                    val_data,
                    test_data,
                    utils.METRICS[config.output_type],
                    model_saver,
                    writer,
                    steps,
                )

            if steps >= config.num_iterations:
                break
    model.eval()
    return model


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_num_threads(5)
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/bfs.yaml")
    parser.add_argument("--num_seeds", type=int, default=3)

    options = parser.parse_args()

    print("Train with config {}".format(options.config_path))

    for seed in range(40, 40 + options.num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = base_config.read_config(options.config_path)
        model = train(config, seed)
