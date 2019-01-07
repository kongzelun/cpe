import argparse
import logging
import time
import torch
import models
import dataset
import numpy as np
from config import Config
from torch import optim
from torch.utils.data import DataLoader

def stream(config, trainset, streamset):
    logger = logging.getLogger(__name__)

    net = models.DenseNet(device=torch.device(config.device),
                          tensor_view=trainset.tensor_view,
                          number_layers=config.number_layers,
                          growth_rate=config.growth_rate,
                          drop_rate=config.drop_rate)
    logger.info("DenseNet Channels: %d", net.channels)

    criterion = models.CPELoss(gamma=config.gamma, tao=config.tao, b=config.b, beta=config.beta, lambda_=config.lambda_)
    optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    prototypes = models.Prototypes(threshold=config.threshold)
    # load saved prototypes
    try:
        prototypes.load(config.prototypes_path)
        logger.info("load prototypes from file '%s'.", config.prototypes_path)
    except FileNotFoundError:
        pass
    logger.info("original prototype count: %d", len(prototypes))

    def train(train_dataset):
        dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1)
        for epoch in range(config.epoch_number):
            logger.info('----------------------------------------------------------------')
            logger.info("epoch: %d", epoch + 1)
            logger.info("threshold: %.4f, gamma: %.4f, tao: %.4f, b: %.4f", config.threshold, config.gamma, config.tao, config.b)
            logger.info("prototypes count before training: %d", len(prototypes))

            net.train()
            for i, (feature, label) in enumerate(dataloader):
                feature, label = feature.to(net.device), label.to(net.device)
                optimizer.zero_grad()
                feature, out = net(feature)
                loss, distance = criterion(feature, out, label, prototypes)
                loss.backward()
                optimizer.step()

                logger.debug("[%d, %d] %7.4f %7.4f", epoch + 1, i + 1, loss.item(), distance)

            logger.info("prototypes count after training: %d", len(prototypes))
            prototypes.update()
            logger.info("prototypes count after update: %d", len(prototypes))
        else:
            net.save(config.net_path)
            logger.info("net has been saved")
            prototypes.save(config.prototypes_path)
            logger.info("prototypes has been saved")

            intra_distances = []
            with torch.no_grad():
                net.eval()
                for i, (feature, label) in enumerate(dataloader):
                    feature, label = feature.to(net.device), label.item()
                    feature, out = net(feature)
                    closest_prototype, distance = prototypes.closest(feature, label)
                    intra_distances.append((label, distance))

            detector = models.Detector(intra_distances, train_dataset.label_set, config.std_coefficient)
            detector.save(config.detector_path)
            logger.info("detector has been saved")

        return detector

    def test():
        pass

    if config.train:
        for period in range(config.period):
            logger.info('----------------------------------------------------------------')
            logger.info("period: %d", period + 1)
            train(trainset)
    else:
        pass


def main(args):
    config = Config(args)
    logger = logging.getLogger(__name__)

    def setup_logger(level=logging.DEBUG, filename=None):
        logger.setLevel(level)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if filename is not None:
            file_handler = logging.FileHandler(filename=filename, mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.debug("logger '%s' has been setup.", __name__)

    setup_logger(level=logging.DEBUG, filename=config.log_path)

    # start adjusting parameters according to dataset
    if config.dataset == 'fm':
        trainset = dataset.FashionMnist(train=True)
        testset = dataset.FashionMnist(train=False)
        parameters = {
            'number_layers': 6,
            'growth_rate': 12,
            'learning_rate': 0.001,
            'drop_rate': 0.2,
            'threshold': 10.0,
            'gamma': 0.1,
            'tao': 20.0,
            'b': 10.0,
            'beta': 1.0,
            'lambda_': 0.001,
            'std_coefficient': 3.0
        }
    elif config.dataset == 'c10':
        trainset = dataset.Cifar10(train=True)
        testset = dataset.Cifar10(train=False)
        parameters = {
            'number_layers': 8,
            'growth_rate': 16,
            'learning_rate': 0.001,
            'drop_rate': 0.2,
            'threshold': 15.0,
            'gamma': 1 / 15.0,
            'tao': 30.0,
            'b': 15.0,
            'beta': 1.0,
            'lambda_': 0.001,
            'std_coefficient': 3.0
        }
    else:
        raise RuntimeError("Dataset not found.")

    config.update(**parameters)
    # end adjusting parameters according to dataset

    logger.info("****************************************************************")
    logger.info("%s", config)
    logger.info("trainset size: %d", len(trainset))
    logger.info("testset size: %d", len(testset))

    start_time = time.time()

    if config.type == 'stream':
        stream(config=config, trainset=trainset, streamset=testset)

    logger.info("-------------------------------- %.3fs --------------------------------", time.time() - start_time)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(prog="CPE")

    argument_group = arg_parser.add_argument_group(title='arguments')
    argument_group.add_argument('-t', '--type', type=str, help="Running type.", choices=['ce', 'cpe', 'stream'], required=True)
    argument_group.add_argument('-d', '--dir', type=str, help="Running directory path.", required=True)
    argument_group.add_argument('--dataset', type=str, help="Dataset.", choices=dataset.DATASETS, required=True)
    argument_group.add_argument('--device', type=str, help="Torch device.", default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    argument_group.add_argument('-c', '--clear', help="Clear running path.", action="store_true")

    training_group = arg_parser.add_argument_group(title='training arguments')
    training_group.add_argument('--train', help="Whether do training process.", action="store_true")
    training_group.add_argument('-p', '--period', type=int, help="Run the whole process for how many times.", default=1)
    training_group.add_argument('-e', '--epoch', type=int, help="Epoch Number.", default=1)

    stream_group = arg_parser.add_argument_group(title='stream arguments')
    stream_group.add_argument('-r', '--rate', type=float, help='Novelty buffer sample rate.', default=0.3)

    parsed_args = arg_parser.parse_args()

    main(parsed_args)