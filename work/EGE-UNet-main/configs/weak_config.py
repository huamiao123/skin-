from datetime import datetime
from torchvision import transforms
from utils import GT_BceDiceLoss, myNormalize, myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip, myRandomRotation, myResize


class WeakBaseConfig:
    network = 'ege_wave_weak'

    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [8, 16, 24, 32, 48, 64],
        'bridge': True,
        'gt_ds': True,
    }

    datasets = 'isic18'
    data_path = './data/isic2018/'
    weak_json_path = './data/isic2018/weak_annotations/weak_train.json'

    criterion = GT_BceDiceLoss(wb=1, wd=1)

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 8
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 64
    epochs = 300

    work_dir = 'results/weak_baseline_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    sch = 'CosineAnnealingLR'
    T_max = 300
    eta_min = 0.00001
    last_epoch = -1

    weak_mode = 'baseline'

    lambda_partial = 1.0
    lambda_box_lf = 1.0
    lambda_local_point = 1.0
    lambda_consistency = 0.5

    train_transformer = transforms.Compose([
        myNormalize('isic18', train=True),
        myToTensor(),
        myResize(256, 256)
    ])
    test_transformer = transforms.Compose([
        myNormalize('isic18', train=False),
        myToTensor(),
        myResize(256, 256)
    ])


class WeakFullConfig(WeakBaseConfig):
    network = 'ege_wave_weak'
    weak_mode = 'full'

    work_dir = 'results/weak_full_isic18_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
