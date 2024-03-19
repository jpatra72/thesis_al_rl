import os

from tools.helper import create_parser_from_yaml


def evaluate_al_agent(args):


    pass


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args = create_parser_from_yaml(default='dqn_test', path=current_dir)
    evaluate_al_agent(args)
