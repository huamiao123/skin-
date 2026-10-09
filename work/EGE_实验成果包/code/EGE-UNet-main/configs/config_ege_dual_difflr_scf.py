from configs.config_ege_dual_difflr import ege_dual_difflr_config
from datetime import datetime


class ege_dual_difflr_scf_config(ege_dual_difflr_config):
    network = 'ege_dual'
    fusion_type = 'scale_cond'

    work_dir = 'results/ege_dual_difflr_scf_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
