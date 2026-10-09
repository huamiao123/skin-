from configs.config_setting import setting_config
from datetime import datetime


class ege_dual_config(setting_config):
    network = 'ege_dual'

    t_embed = 48
    t_depths = (2, 2, 2, 2)
    t_head_dim = 16
    t_sr_ratios = (4, 2, 1, 1)
    t_mlp_ratio = 4.0
    t_drop_path_rate = 0.1

    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    work_dir = 'results/ege_dual_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
