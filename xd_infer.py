import torch
import argparse
from xd_utils import valid, get_xd_test_dataloader

from model.model import FadNet


def parse_args():
    parser = argparse.ArgumentParser(description='FADNet gogogo!')
    parser.add_argument('--num_iters', type=int, default=2000)
    parser.add_argument('--len_feature', type=int, default=1024)
    parser.add_argument('--output_path', type=str, default='experiment/')
    parser.add_argument('--root_dir', type=str, default='outputs/')
    parser.add_argument('--modal', type=str, default='rgb', choices=["rgb,flow,both"])
    parser.add_argument('--model_path', type=str, default='weights/')
    parser.add_argument('--lr', type=str, default=0.0001, help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_segments', type=int, default=32)
    parser.add_argument('--worker_init_fn', default=None)
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_loader = get_xd_test_dataloader(args)
    net = FadNet(args.len_feature, flag="Test", a_nums=60, n_nums=60)

    torch.cuda.set_device(0)
    net = net.cuda()

    path = 'set your model path'
    valid(net, test_loader, model_file=path)
