import os
import time
import torch
import random
import numpy as np
from utils import args_util, io_util, model_util, pipeline


def load_embedding(path):
    embeds = {}
    for line in open(path, encoding='utf-8'):
        elements = line.strip().split()
        if len(elements) == 2:
            continue
        vector = [float(i) for i in elements[1:]]
        token = elements[0]
        if token == '<unk>':
            token = '[UNK]'
        embeds[token] = vector
    return embeds


def main():
    arg_parser = args_util.add_general_args()
    arg_parser = args_util.add_train_args(arg_parser)
    arg_parser = args_util.add_model_args(arg_parser)
    args = arg_parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if torch.cuda.is_available() and args.gpu_id > -1:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    config = {'root_folder': args.root_folder,
              'author_dim': args.author_dim,
              'author_track_dim': args.author_track_dim,
              'topic_dim': args.topic_dim,
              'token_dim': args.token_dim,
              'rnn_type': args.rnn_type,
              'rnn_layer': args.rnn_layer,
              'hid_dim': args.hid_dim,
              'dropout': args.dropout,
              'sentiment_dim': args.sentiment_dim,
              'emotion_dim': args.emotion_dim,
              'build_sentiment_embedding': args.build_sentiment_embedding,
              'build_author_emb': args.build_author_emb,
              'build_author_track': args.build_author_track,
              'build_author_predict': args.build_author_predict,
              'build_topic_predict': args.build_topic_predict,
              'build_sentiment_predict': args.build_sentiment_predict,
              'build_emotion_predict': args.build_emotion_predict,
              'leverage_topic': args.leverage_topic,
              'leverage_emotion': args.leverage_emotion,
              'lr': args.lr,
              'epoch': args.epoch,
              'update_iter': args.update_iter,
              'grad_clip': args.grad_clip,
              'batch_size': args.batch_size,
              'check_step': args.check_step,
              'random_seed': args.random_seed,
              'previous_comment_cnt': args.previous_comment_cnt,
              'min_comment_cnt': args.min_comment_cnt,
              'max_seq_len': args.max_seq_len,
              'build_auxiliary_task': args.build_auxiliary_task,
              'detach_article': args.detach_article,
              'track_max_pool': args.track_max_pool,
              'track_mean_pool': args.track_mean_pool,
              'track_last_pool': args.track_last_pool,
              'token_max_pool': args.token_max_pool,
              'token_mean_pool': args.token_mean_pool,
              'token_last_pool': args.token_last_pool,
              'free_fp': args.free_fp,
              'freeze_aux': args.freeze_aux,
              'track_grad': args.track_grad,
              'vader': args.vader,
              'flair': args.flair,
              'sent': args.sent,
              'subj': args.subj,
              'emotion': args.emotion,
              }
    for key, value in config.items():
        print(key, value)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if args.embedding_weight and os.path.isfile(args.embedding_weight):
        embedding = load_embedding(args.embedding_weight)
    else:
        embedding = None
    for outlet in ['Archiveis', 'wsj', 'NewYorkTimes']:  # os.listdir(args.root_folder):
        print("Working on {} ...".format(outlet))
        io = io_util.IO(folder_path=os.path.join(args.root_folder, outlet),
                        batch_size=config['batch_size'],
                        max_seq_len=config['max_seq_len'],
                        previous_comment_cnt=config['previous_comment_cnt'],
                        min_comment_cnt=config['min_comment_cnt'],
                        target_sentiment=True,
                        target_emotion=config['emotion'])
        config['author_size'] = len(io.authors)
        config['topic_size'] = io.topic_size
        config['token_size'] = len(io.word2idx)
        config['outlet'] = outlet

        model = model_util.Model(config)
        sgd = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)
        # sgd = torch.optim.SGD(model.parameters(), lr=config['lr'])
        # sgd = torch.optim.RMSprop(model.parameters(), centered=True)
        if os.path.isfile(os.path.join(config['root_folder'],
                                       config['outlet'], 'best_model.pt')) and args.load_checkpoint:
            checkpoint = torch.load(os.path.join(outlet, args.model_file))
            model.to(device)
            model.load_state_dict(checkpoint['model'])
            sgd.load_state_dict(checkpoint['sgd'])
        else:
            model.build_embedding(vocab=io.word2idx, embedding=embedding)
            model.to(device)
        pipe = pipeline.Pipeline(data_io=io,
                                 epoch=args.epoch,
                                 model=model,
                                 sgd=sgd,
                                 config=config,
                                 device=device)
        pipe.run()

        print("Finish at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == '__main__':
    main()
    # load_embedding('d:/data/embedding/en.wiki.bpe.vs25000.d100.w2v.txt')
