from configs.config_setting import setting_config
from datetime import datetime


class wave_ll_only_config(setting_config):
    network = 'ege_wave_unet'
    wave_mode = 'll_only'
    fusion_type = 'scalar'

    work_dir = 'results/ege_wave_ll_only_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
