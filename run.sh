python main.py --random_seed 126 \
--root_folder /home/fanyang/fYYw/code/code_python/data/news/outlets \
--previous_comment_cnt 16 \
--max_seq_len 128 \
--embedding_weight /home/fanyang/fYYw/code/code_python/data/news/outlets/en.wiki.bpe.vs25000.d300.w2v.txt \
--rnn_type lstm \
--hid_dim 1024 \
--token_dim 300 \
--dropout 0 \
--rnn_layer 2 \
--author_dim 64 \
--author_track_dim 1024 \
--topic_dim 64 \
--emotion_dim 6 \
--sentiment_dim 64 \
--build_sentiment_embedding true \
--build_author_emb false \
--build_author_track true \
--build_author_predict true \
--build_topic_predict true \
--leverage_topic true \
--leverage_emotion true \
--sentiment_fingerprinting true \
--emotion_fingerprinting true \
--lr 0.001 \
--epoch 60 \
--update_iter 1 \
--grad_clip 1 \
--use_entire_example_epoch 8 \
--batch_size 256 \
--update_size 1 \
--check_step 100 \
--load_checkpoint false