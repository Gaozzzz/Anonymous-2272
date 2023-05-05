import os
import torch
import argparse
from tqdm import tqdm
from loss_function import AD_Loss, AD_rtfm_Loss
from ucf_utils import test, get_ucf_dataloader
from utils import train, save_experiment_results

from model.model import FadNet


def parse_args():
    parser = argparse.ArgumentParser(description='FADNet gogogo!')
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--len_feature', type=int, default=1024)
    parser.add_argument('--memory_block_number', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_segments', type=int, default=32)
    parser.add_argument('--worker_init_fn', default=None)
    parser.add_argument('--lr', type=str, default=0.0001, help='learning rates for steps(list form)')

    parser.add_argument('--root_dir', type=str, default='outputs/')
    parser.add_argument('--model_path', type=str, default='weights/')
    parser.add_argument('--output_path', type=str, default='experiment/')
    parser.add_argument('--modal', type=str, default='rgb', choices=["rgb,flow,both"])
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.memory_block_number = 60
    net = FadNet(args.len_feature, flag="Train", a_nums=args.memory_block_number, n_nums=args.memory_block_number)
    experiment_name = 'model'
    run_number = 'model_1'
    Cuda = 0
    torch.cuda.set_device(Cuda)  # set your gpu device
    net = net.cuda()
    normal_train_loader, abnormal_train_loader, test_loader = get_ucf_dataloader(args)

    test_info = {"step": [], "auc": [], "ap": [], "ac": []}

    best_auc = 0
    criterion = AD_Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.00005)

    for step in tqdm(
            range(1, args.num_iters + 1),
            total=args.num_iters,
            dynamic_ncols=True
    ):
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)
        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)

        loss = train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion)
        if step % 10 == 0 and step > 10:
            test(net, test_loader, test_info, step)
            print(loss)
            if test_info["auc"][-1] > best_auc:
                print('step:{};auc:{};ap:{}'.format(test_info["step"][-1], test_info["auc"][-1], test_info["ap"][-1]))
                best_auc = test_info["auc"][-1]
                save_experiment_results(experiment_name=experiment_name,
                                        run_number=run_number,
                                        result1=test_info['auc'][-1],
                                        result2=test_info['ap'][-1],
                                        result3=test_info['ac'][-1],
                                        filename=os.path.join(args.output_path, 'ucf_experiment_results.txt'))

                torch.save(net.state_dict(), os.path.join(args.model_path, \
                                                          f'ucf_{experiment_name}_{run_number}_best.pkl'))

            if step == args.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                                                          f'ucf_{experiment_name}_{run_number}_final.pkl'))
