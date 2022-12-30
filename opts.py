import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--start_from', type=str, default='./ckpt/')

    # AoA settings
    parser.add_argument('--ctx_drop', type=int, default=0, help='apply dropout to the context vector before fed into LSTM?')
    parser.add_argument('--use_warmup', type=int, default=0, help='warm up the learning rate?')
    parser.add_argument('--acc_steps', type=int, default=1, help='accumulation steps')
    parser.add_argument('--norm_att_feat', type=int, default=0, help='If normalize attention features')

    # Optimization: General
    parser.add_argument('--max_epoch', type=int, default=40, help='number of epochs')
    parser.add_argument('--loss_amp', type=float, default=1., help='loss amplifier')
    parser.add_argument('--batch_size', type=int, default=200, help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0., help='strength of dropout in the Language Model RNN')

    # Early Detection Parameters
    parser.add_argument('--illicit_name', type=str, default='Ransomware', help='which illicit to detect')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for experiment')
    parser.add_argument('--split_number', type=int, default=23, help='number of splits')
    parser.add_argument('--analysis_span', type=int, default=23, help='number of splits')

    # General settings
    parser.add_argument('--AF_dim', type=int, default=16, help='dim of LT feature we prepare 23 total')
    parser.add_argument('--path_feat_dim', type=int, default=16, help='dim of path feature')
    parser.add_argument('--intersect_dim', type=int, default=10, help='dim of intersection link')
    parser.add_argument('--addr_graph_dim', type=int, default=10, help='dim of address graph node')
    parser.add_argument('--hidden_dim', type=int, default=32, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--path_zoom_len', type=int, default=6, help='zoom into a fixed length path')
    parser.add_argument('--logit_layers', type=int, default=1, help='number of layers in the RNN')

    # Evaluation/Checkpointing
    parser.add_argument('--save_checkpoint_every', type=int, default=15, help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--losses_log_every', type=int, default=10, help='how often to save log')
    parser.add_argument('--save_history_ckpt', type=int, default=1, help='If save checkpoints at every save point')
    parser.add_argument('--checkpoint_path', type=str, default='./saved_ckpt', help='directory to store checkpointed models')
    parser.add_argument('--load_best_score', type=int, default=1, help='Do we load previous best score when resuming training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=3, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=1, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.75, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--skip_thres_1', type=float, default=1., help='')
    parser.add_argument('--skip_thres_2', type=float, default=-1., help='')

    args, unknown = parser.parse_known_args()
    return args
