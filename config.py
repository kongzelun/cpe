import os
import shutil
import json


class Config:
    def __init__(self, args):
        # read from args
        self.path = "{}/config.json".format(args.dir)
        self.running_path = args.dir
        self.clear = args.clear
        self.type = args.type
        self.train = args.train
        self.period = args.period if args.train else 1
        self.epoch_number = args.epoch if args.train else 1
        self.dataset = args.dataset
        self.device = args.device
        self.novelty_buffer_sample_rate = args.rate if self.type == 'stream' else 0.0

        # derived
        self.parameter_path = os.path.join(self.running_path, "parameter.json")
        self.log_path = os.path.join(self.running_path, "run.log")
        self.net_path = os.path.join(self.running_path, "model.pkl")
        self.prototypes_path = os.path.join(self.running_path, "prototypes.pkl")
        self.detector_path = os.path.join(self.running_path, "detector.pkl")
        self.probs_path = os.path.join(self.running_path, "probs.pkl")

        # parameters
        if self.dataset == 'fm':
            self.number_layers = 6
            self.growth_rate = 12
            self.learning_rate = 0.001
            self.drop_rate = 0.2
            self.threshold = 10.0
            self.gamma = 0.1
            self.tao = 20.0
            self.b = 10.0
            self.beta = 1.0
            self.lambda_ = 0.001
            self.std_coefficient = 3.0
        elif self.dataset == 'c10':
            self.number_layers = 8
            self.growth_rate = 16
            self.learning_rate = 0.001
            self.drop_rate = 0.2
            self.threshold = 15.0
            self.gamma = 1 / 15.0
            self.tao = 30.0
            self.b = 15.0
            self.beta = 1.0
            self.lambda_ = 0.001
            self.std_coefficient = 1.0

        if not os.path.isdir(self.running_path):
            os.mkdir(self.running_path)

        if self.clear:
            shutil.rmtree(args.dir)
            os.mkdir(args.dir)

        if os.path.isfile(self.path):
            with open(self.path) as file:
                config_dict = json.load(file)
            self.update(**config_dict)

        self.dump()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        self.dump()

    def dump(self):
        with open(self.parameter_path, 'w') as file:
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
        s = ''
        for k, v in self.__dict__.items():
            s = s + '\n{}: {}'.format(k, v)

        return s
