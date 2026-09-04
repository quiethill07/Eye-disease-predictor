import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate split-safe train/test segmentation masks for segmentation-guided classification.'
    )
    parser.add_argument('--name', required=True, type=str, help='segmentation experiment name in output_dir')
    parser.add_argument('--output_dir', default='outputs', type=str, help='segmentation output root')
    parser.add_argument('--data_dir', required=True, type=str, help='FIVES root containing train/test folders')
    parser.add_argument('--save_root', default='classifier_masks', type=str, help='where generated masks are stored')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--mask_type', default='prob', choices=['prob', 'binary'],
                        help='type of masks to generate for classifier guidance')
    parser.add_argument('--restore_size', default=True, action='store_true', help='restore masks to original size')
    parser.add_argument('--no_restore_size', dest='restore_size', action='store_false')
    parser.set_defaults(restore_size=True)
    return parser.parse_args()


def run_cmd(cmd, cwd):
    print('\n>>', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_img_dir = os.path.join(args.data_dir, 'train', 'Original')
    test_img_dir = os.path.join(args.data_dir, 'test', 'Original')
    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(f'Missing train image dir: {train_img_dir}')
    if not os.path.isdir(test_img_dir):
        raise FileNotFoundError(f'Missing test image dir: {test_img_dir}')

    train_save_dir = os.path.join(args.save_root, args.name, 'train')
    test_save_dir = os.path.join(args.save_root, args.name, 'test')
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    base_cmd = [
        sys.executable,
        'infer.py',
        '--name', args.name,
        '--output_dir', args.output_dir,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--threshold', str(args.threshold),
        '--manifest_name', 'manifest.csv',
        '--restore_size', 'true' if args.restore_size else 'false',
    ]
    if args.mask_type == 'prob':
        base_cmd += [
            '--save_binary', 'false',
            '--save_prob', 'true',
            '--prob_subdir', 'prob_maps',
        ]
        target_subdir = 'prob_maps'
    else:
        base_cmd += [
            '--save_binary', 'true',
            '--save_prob', 'false',
            '--binary_subdir', 'binary_masks',
        ]
        target_subdir = 'binary_masks'

    train_cmd = base_cmd + [
        '--image_dir', train_img_dir,
        '--save_dir', train_save_dir,
        '--id_prefix', 'fives_train',
    ]
    test_cmd = base_cmd + [
        '--image_dir', test_img_dir,
        '--save_dir', test_save_dir,
        '--id_prefix', 'fives_test',
    ]

    run_cmd(train_cmd, cwd=script_dir)
    run_cmd(test_cmd, cwd=script_dir)

    print('\nGenerated mask folders:')
    print(f'- Mask type  : {args.mask_type}')
    print(f'- Train masks: {os.path.join(train_save_dir, target_subdir)}')
    print(f'- Test masks : {os.path.join(test_save_dir, target_subdir)}')


if __name__ == '__main__':
    main()
