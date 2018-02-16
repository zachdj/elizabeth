import argparse

import elizabeth


def info(args):
    '''Print system info.
    '''
    import sys
    print('Python version:')
    print(sys.version)


def main():
    parser = argparse.ArgumentParser(
        description='Scalable malware detection',
        argument_default=argparse.SUPPRESS,
    )
    subcommands = parser.add_subparsers()

    # elizabeth info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    # elizabeth nb <train_x> <train_y> <test_x> [<test_y>]
    cmd = subcommands.add_parser('nb', description='naive bayes', argument_default=argparse.SUPPRESS)
    cmd.add_argument('train_x', help='path to the training set')
    cmd.add_argument('train_y', help='path to the training labels')
    cmd.add_argument('test_x', help='path to the test set')
    cmd.add_argument('test_y', help='path to the test labels', nargs='?')
    cmd.add_argument('--ngram', help='the ngram size')
    cmd.add_argument('--idf', help='use TF-IDF rather than plain TF', action='store_true')
    cmd.add_argument('--asm', help='use assembly opcodes instead of bytes', action='store_true')
    cmd.add_argument('--base', help='base url of the data files', default='gs')
    cmd.set_defaults(func=elizabeth.naive_bayes.main)

    # elizabeth rf <train_x> <train_y> <test_x> [<test_y>]
    cmd = subcommands.add_parser('rf', description='random forest', argument_default=argparse.SUPPRESS)
    cmd.add_argument('train_x', help='path to the training set')
    cmd.add_argument('train_y', help='path to the training labels')
    cmd.add_argument('test_x', help='path to the test set')
    cmd.add_argument('test_y', help='path to the test labels', nargs='?')
    cmd.add_argument('--base', help='base url of the data files', default='gs')
    cmd.set_defaults(func=elizabeth.tree_ensemble.main)

    # Each subcommand gives an `args.func`.
    # Call that function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
