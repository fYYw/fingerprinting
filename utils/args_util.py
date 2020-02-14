import argparse


def str2bool(v):
    if v.lower() in ['yes', 'y', 1, 'true', 't']:
        return True
    elif v.lower() in ['no', 'n', 0, 'false', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def add_general_args(args=None):
    args = args if args else argparse.ArgumentParser()
    args.add_argument('--random_seed', type=int, default=126)
    args.add_argument('--gpu_id', type=int, default=0)
    args.add_argument('--root_folder', type=str, default='D:/data/outlets')
    args.add_argument('--previous_comment_cnt', type=int, default=12)
    args.add_argument('--max_seq_len', type=int, default=128)
    args.add_argument('--prob_to_full', type=float, default=1.)
    args.add_argument('--embedding_weight', default='d:/data/embedding/en.wiki.bpe.vs25000.d300.w2v.txt')
    # args.add_argument('--embedding_weight', default='')
    return args


def add_model_args(args=None):
    args = args if args else argparse.ArgumentParser()
    args.add_argument('--rnn_type', default='lstm', choices=['lstm', 'gru'])
    args.add_argument('--hid_dim', type=int, default=256)
    args.add_argument('--token_dim', type=int, default=300)
    args.add_argument('--dropout', type=float, default=0.)
    args.add_argument('--rnn_layer', type=int, default=2)
    args.add_argument('--author_dim', type=int, default=64)
    args.add_argument('--author_track_dim', type=int, default=256)
    args.add_argument('--topic_dim', type=int, default=64)
    args.add_argument('--emotion_dim', type=int, default=6)
    args.add_argument('--sentiment_dim', type=int, default=64)
    args.add_argument('--build_sentiment_embedding', type=str2bool, default=True)
    args.add_argument('--build_author_emb', type=str2bool, default=False)
    args.add_argument('--build_author_track', type=str2bool, default=True)
    args.add_argument('--build_author_predict', type=str2bool, default=True)
    args.add_argument('--build_topic_predict', type=str2bool, default=True)
    args.add_argument('--leverage_topic', type=str2bool, default=True)
    args.add_argument('--leverage_emotion', type=str2bool, default=True)
    args.add_argument('--sentiment_fingerprinting', type=str2bool, default=True)
    args.add_argument('--emotion_fingerprinting', type=str2bool, default=True)
    return args


def add_train_args(args=None):
    args = args if args else argparse.ArgumentParser()
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--epoch', type=int, default=40)
    args.add_argument('--update_iter', type=int, default=1)
    args.add_argument('--grad_clip', type=float, default=1.)
    args.add_argument('--check_epoch', type=int, default=-1)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--update_size', type=int, default=1, help='Backward() without gradient step.')
    args.add_argument('--check_step', type=int, default=100, help='Validate every # steps. ')
    args.add_argument('--load_checkpoint', type=str2bool, default=False)
    return args


if __name__ == '__main__':
    """ Test """
    print()