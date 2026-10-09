from configs.config_setting import setting_config
from datetime import datetime


class psudr_config(setting_config):
    network = 'ege_wave_psudr'

    work_dir = 'results/ege_wave_psudr_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
