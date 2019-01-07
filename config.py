import os
import shutil
import json

class Config:
    # read from args
    type = None
    clear = False
    train = True
    running_path = None
    period = 1
    epoch_number = 1
    device = 'cpu'
    dataset = None
    novelty_buffer_sample_rate = 0.3

    # read from json
    number_layers = 6
    growth_rate = 12
    learning_rate = 0.001
    drop_rate = 0.1,
    threshold = 10.0
    gamma = 0.1
    tao = 10.0
    b = 1.0
    beta = 0.1
    lambda_ = 0.1
    std_coefficient = 1.0

    # derived
    path = None
    log_path = None
    optim_path = None
    net_path = None
    prototypes_path = None
    detector_path = None
    probs_path = None

    def __init__(self, args):
        self.path = "{}/config.json".format(args.dir)

        if not os.path.isdir(args.dir):
            os.mkdir(args.dir)

        if os.path.isfile(self.path):
            with open(self.path) as file:
                config_dict = json.load(file)
            self.update(**config_dict)

        if args.clear:
            shutil.rmtree(args.dir)
            os.mkdir(args.dir)
            self.clear = True

        self.type = args.type
        self.running_path = args.dir
        self.dataset = args.dataset
        self.device = args.device

        self.train = args.train
        self.period = args.period if args.train else 1
        self.epoch_number = args.epoch if args.train else 1

        if self.type == 'stream':
            self.novelty_buffer_sample_rate = args.rate

        self.log_path = os.path.join(self.running_path, "run.log")
        self.net_path = os.path.join(self.running_path, "model.pkl")
        self.prototypes_path = os.path.join(self.running_path, "prototypes.pkl")
        self.detector_path = os.path.join(self.running_path, "detector.pkl")
        self.probs_path = os.path.join(self.running_path, "probs.pkl")

        self.dump()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def dump(self):
        with open(self.path, 'w') as file:
            parameters = {
                'number_layers': self.number_layers,
                'growth_rate': self.growth_rate,
                'learning_rate': self.learning_rate,
                'drop_rate': self.drop_rate,
                'threshold': self.threshold,
                'gamma': self.gamma,
                'tao': self.tao,
                'b': self.b,
                'beta': self.beta,
                'lambda_': self.lambda_,
                'std_coefficient': self.std_coefficient
            }
            json.dump(parameters, file)

    def __repr__(self):
        return "{}".format(self.__dict__)
