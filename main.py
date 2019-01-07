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
    logger.info("densenet channel: %d", net.channels)

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

    detector = None

    def train(train_dataset):
        dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
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
            logger.info("net has been saved.")
            prototypes.save(config.prototypes_path)
            logger.info("prototypes has been saved.")

            intra_distances = []
            with torch.no_grad():
                net.eval()
                for i, (feature, label) in enumerate(dataloader):
                    feature, label = feature.to(net.device), label.item()
                    feature, out = net(feature)
                    closest_prototype, distance = prototypes.closest(feature, label)
                    intra_distances.append((label, distance))

            novelty_detector = models.Detector(intra_distances, prototypes.label_set, config.std_coefficient)
            logger.info("distance average: %s", detector.average_distances)
            logger.info("distance std: %s", detector.std_distances)
            logger.info("detector threshold: %s", detector.thresholds)
            detector.save(config.detector_path)
            logger.info("detector has been saved.")

        return novelty_detector

    def test(test_dataset, novelty_detector, known_labels):
        dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        novelty_detector.known_labels = known_labels

        detection_results = []

        with torch.no_grad():
            net.eval()
            for i, (feature, label) in enumerate(dataloader):
                feature, label = feature.to(net.device), label.item()
                feature, out = net(feature)
                predicted_label, distance = models.predict(feature, prototypes)
                prob = models.probability(feature, predicted_label, prototypes, gamma=config.gamma)
                detected_novelty = detector(predicted_label, distance)
                real_novelty = label not in novelty_detector.known_labels

                detection_results.append((label, predicted_label, real_novelty, detected_novelty))

                logger.debug("[test %5d]: %d, %d, %7.4f, %7.4f, %5s, %5s",
                             i + 1, label, predicted_label, prob, distance, real_novelty, detected_novelty)

    def stream_train(train_dataset, stream_dataset):
        novelty_detector = train(trainset)

        novelty_dataset = dataset.NoveltyDataset(train_dataset)
        iter_streamloader = enumerate(DataLoader(dataset=stream_dataset, batch_size=1, shuffle=True))
        buffer = []

        for i, (feature, label) in iter_streamloader:
            sample = (feature.squeeze(dim=0), label.squeeze(dim=0))
            with torch.no_grad():
                net.eval()
                feature, label = feature.to(net.device), label.item()
                feature, out = net(feature)
                predicted_label, distance = models.predict(feature, prototypes)
                prob = models.probability(feature, predicted_label, prototypes, gamma=config.gamma)
                detected_novelty = novelty_detector(predicted_label, distance)
                real_novelty = label not in detector.known_labels

            if detected_novelty:
                buffer.append(sample)

            logger.debug("[stream %5d]: %d, %d, %7.4f, %7.4f, %5s, %5s, %4d",
                         i + 1, label, predicted_label, prob, distance, real_novelty, detected_novelty, len(buffer))

            if len(buffer) >= 1000:
                logger.info("novelty dataset size before extending: %d", len(novelty_dataset))
                novelty_dataset.extend(buffer, config.novelty_buffer_sample_rate)
                logger.info("novelty dataset size after extending: %d", len(novelty_dataset))
                novelty_detector = train(novelty_dataset)
                buffer.clear()

        return novelty_detector

    if config.train:
        for period in range(config.period):
            logger.info('----------------------------------------------------------------')
            logger.info("period: %d", period + 1)
            logger.info("stream training started.")
            detector = stream_train(trainset, streamset)

        logger.info("testing started.")
        test(streamset, detector, trainset.label_set)
    else:
        logger.info("testing started.")
        # test(streamset, detector, trainset.label_set)


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

    if config.dataset == 'fm':
        trainset = dataset.FashionMnist(train=True)
        testset = dataset.FashionMnist(train=False)
    elif config.dataset == 'c10':
        trainset = dataset.Cifar10(train=True)
        testset = dataset.Cifar10(train=False)
    else:
        raise RuntimeError("Dataset not found.")

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
