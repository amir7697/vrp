import random
import torch
import argparse

import dataaccess.simulation as data_access

from model.vrp_model import VrpModel
from model.vrp_model_supervisor import VrpModelSupervisor
from dataaccess.data_processor import VrpDataProcessor
from config.configs import SIMULATION_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG, TRAIN_CONFIG


def get_arguments(title):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--eval', action='store_true')

    params = parser.parse_args()

    params = dict(params.items() + SIMULATION_CONFIG.items() + MODEL_CONFIG.items() +
                  OUTPUT_CONFIG.items() + TRAIN_CONFIG.items())
    return params


def create_model(args):
    model = VrpModel(args)
    model_supervisor = VrpModelSupervisor(model, args)
    if args.pre_trained_model:
        print('Using pre trained model in {}'.format(args.pre_trained_model_path))
        model_supervisor.load_pre_trained_model(args.pre_trained_model_path)
    else:
        print('Created model with fresh parameters.')
        model_supervisor.model.init_weights(args.param_init)

    return model_supervisor


def train(args):
    print('Training starts.')

    train_data = data_access.load_data_set(args.train_data_path)
    train_data_size = len(train_data)
    evaluation_data = data_access.load_data_set(args.evaluation_data_path)
    model_supervisor = create_model(args)
    data_processor = VrpDataProcessor()

    start_idx = args.resume_idx * args.batch_size

    for epoch in range(start_idx//train_data_size, args.num_epoch):
        random.shuffle(train_data)
        for batch_idx in range(start_idx%train_data_size, train_data_size, args.batch_size):
            print("epoch: {}, batch_idx: {}".format(epoch, batch_idx))
            batch_data = data_processor.get_batch(train_data, args.batch_size, batch_idx)
            train_loss, train_reward = model_supervisor.train(batch_data)
            print('train loss: %.4f train reward: %.4f' % (train_loss, train_reward))

            if model_supervisor.global_step % args.evaluation_step == 0:
                eval_loss, eval_reward = model_supervisor.eval(evaluation_data, args.output_trace_flag, args.max_eval_size)
                val_summary = {'avg_reward': eval_reward, 'global_step': model_supervisor.global_step}
                print('global step: {}'.format(val_summary['global_step']))
                print('average reward: {}'.format((val_summary['avg_reward'])))
        start_idx = 0


def evaluate(args):
    print('Evaluation starts.')

    test_data = data_access.load_data_set(args.test_data_path)
    args.dropout_rate = 0.0
    model_supervisor = create_model(args)
    test_loss, test_reward = model_supervisor.eval(test_data, args.output_trace_flag)
    print('test loss: %.4f test reward: %.4f' % (test_loss, test_reward))


if __name__ == '__main__':
    arguments = get_arguments("vrp")
    arguments.cuda = not arguments.cpu and torch.cuda.is_available()

    if arguments.train:
        train(arguments)
    else:
        evaluate(arguments)