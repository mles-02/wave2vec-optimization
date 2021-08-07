import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    # FIXME
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')

    # FIXME
    # Do Training

