from configs.config_setting import setting_config
from datetime import datetime


class prh_config(setting_config):
    network = 'ege_prh'

    edge_pos_weight = 14.4805

    work_dir = 'results/ege_prh_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
