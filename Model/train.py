from argparse import ArgumentParser
import yaml
from utils import dict2namespace
from main import Handler

if __name__ == '__main__':
    parser = ArgumentParser(description='Efficient Network')

    parser.add_argument('--config', type=str, default='gray.yml')
    parser.add_argument('--useCPU', action='store_true', default=False)
    parser.add_argument('--useAMP', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()

    with open('Model/configs/{}'.format(args.config), 'r') as f:
        configs = yaml.safe_load(f)
    args = dict2namespace(configs, args)
    if args.seed != -1:
        args.train.seed = args.seed

    runner = Handler(args)
    runner.train()