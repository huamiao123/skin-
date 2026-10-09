from configs.config_setting import setting_config
from datetime import datetime


class nohalting_config(setting_config):
    network = 'ege_hrvit_nohalting'

    work_dir = 'results/ege_hrvit_nohalting_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
