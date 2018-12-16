import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from datetime import datetime
import zipfile
import json
import utils

def zipdir(path, ziph):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".py") or file.endswith("cfg"):
            ziph.write(os.path.join(path, file))
            if file.endswith("cfg"):
                os.remove(file)

def save_config(config):
    CONFIG_DIR = './config'
    utils.mkdir(CONFIG_DIR)
    current_time = str(datetime.now()).replace(":", "_")
    save_name = os.path.join(CONFIG_DIR, "txt2img_files_{}.{}")
    with open(save_name.format(current_time, "cfg"), "w") as f:
        for k, v in sorted(args.items()):
            f.write('%s: %s\n' % (str(k), str(v)))

    zipf = zipfile.ZipFile(save_name.format(
        current_time, "zip"), 'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)
    mkdir(config.sample_path)
    mkdir(config.test_path)

    # Load Vocab
    vocab = json.load(open(config.vocab_json, 'r'))

    # Data Loaders
    H, W = config.image_size
    train_loader = get_loader(vocab, config.images_path, config.train_h5, config.batch_size,
                              config.train_masks, config.mask_size, image_size = (H, W), mode = 'train')
    test_loader = get_loader(vocab, config.images_path, config.test_h5, config.batch_size, 
                              config.test_masks, config.mask_size, image_size = (H, W), mode = 'test')

    # Solver
    solver = Solver(vocab, train_loader, test_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
    	pass
    	solver.test(config.test_batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Variables
    VG_DIR = 'vg'
    
    # Scene Graph Model Generator
    parser.add_argument('--mask_size', default=16, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--gconv_dim', default=128, type=int)
    parser.add_argument('--gconv_hidden_dim', default=512, type=int)
    parser.add_argument('--gconv_num_layers', default=5, type=int)
    parser.add_argument('--mlp_normalization', default='none', type=str)
    parser.add_argument('--normalization', default='batch')
    parser.add_argument('--activation', default='leakyrelu-0.2')
    parser.add_argument('--layout_noise_dim', default=32, type=int)
    parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

    # Refinement for generator
    parser.add_argument('--netG', type=str, default='global')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsample_global', type=int, default=2)
    parser.add_argument('--n_blocks_global', type=int, default=2)
    parser.add_argument('--n_blocks_local', type=int, default=3)
    parser.add_argument('--n_local_enhancers', type=int, default=1)
    parser.add_argument('--niter_fix_global', type=int, default=0)

    # Generator losses
    parser.add_argument('--mask_loss_weight', default=2.0, type=float)
    parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
    parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
    parser.add_argument('--predicate_pred_loss_weight',
                        default=0, type=float)  # DEPRECATED

    # Scene Graph Model Discriminators
    # Generic discriminator options
    parser.add_argument('--discriminator_loss_weight',
                        default=0.01, type=float)
    parser.add_argument('--gan_loss_type', default='multi_lsgan')
    parser.add_argument('--d_clip', default=None, type=float)
    parser.add_argument('--d_normalization', default='batch')
    parser.add_argument('--d_padding', default='valid')
    parser.add_argument('--d_activation', default='leakyrelu-0.2')

    # Object discriminator
    parser.add_argument('--crop_size', default=32, type=int)
    parser.add_argument('--d_obj_weight', default=1.0,
                        type=float)  # multiplied by d_loss_weight
    parser.add_argument('--ac_loss_weight', default=0.1, type=float)

    # Discriminators
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    # Image Discriminator
    parser.add_argument('--num_D_img', type=int, default=2, help='number of img discriminators to use')
    parser.add_argument('--n_layers_D_img', type=int, default=1, help='only used if which_model_netD==n_layers')
    # Object Discriminator
    parser.add_argument('--num_D_obj', type=int, default=2, help='number of obj discriminators to use')
    parser.add_argument('--n_layers_D_obj', type=int, default=1, help='only used if which_model_netD==n_layers')

    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--d_img_weight', default=1.0,
                        type=float)  # multiplied by d_loss_weight

    # Dataset options common to both VG and COCO
    parser.add_argument('--image_size', default='64,64', type=int_tuple)
    parser.add_argument('--num_train_samples', default=None, type=int)
    parser.add_argument('--num_val_samples', default=1024, type=int)
    parser.add_argument('--shuffle_val', default=True, type=bool_flag)
    parser.add_argument('--include_relationships',
                        default=True, type=bool_flag)

    # VG-specific options
    parser.add_argument(
        '--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
    parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'test.h5'))
    parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
    parser.add_argument('--test_h5', default=os.path.join(VG_DIR, 'test.h5'))
    parser.add_argument(
        '--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
    parser.add_argument(
        '--train_masks', default=os.path.join(VG_DIR, 'test_masks2.json'))
    parser.add_argument(
        '--test_masks', default=os.path.join(VG_DIR, 'test_masks2.json'))
    parser.add_argument('--max_objects_per_image', default=10, type=int)
    parser.add_argument('--vg_use_orphaned_objects',
                        default=True, type=bool_flag)

    # Training settings
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--init_epochs', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=300)
    # parser.add_argument('--init_iterations', type=int, default=0)
    # parser.add_argument('--num_iterations', type=int, default=1000000)
    parser.add_argument('--eval_mode_after', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--test_batch', type=int, default=5)

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join(VG_DIR, 'images'))
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--test_path', type=str, default='./test samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=25000)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    save_config(config)
    main(config)
