import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm

from dataloader import XDVideo


def test(net, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("Data/xd_gt.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        for i in range(len(test_loader.dataset) // 5):
            _data, _label = next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()
            cls_label.append(int(_label[0]))
            res = net(_data)

            a_predict = res["frame"].cpu().numpy().mean(0)
            cls_pre.append(1 if a_predict.max() > 0.5 else 0)
            fpre_ = np.repeat(a_predict, 16)
            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)


def valid(net, test_loader, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            # net.load_state_dict(torch.load(model_file))
            pretrained_dict = torch.load(model_file, map_location='cuda:0')
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

        pre_dict = {}
        gt_dict = {}
        load_iter = iter(test_loader)
        frame_gt = np.load("Data/xd_gt.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        count = 0
        for i in tqdm(range(len(test_loader.dataset) // 5)):

            _data, _label = next(load_iter)

            _data = _data.cuda()
            _label = _label.cuda()

            cls_label.append(int(_label[0]))
            res = net(_data)
            a_predict = res["frame"].cpu().numpy().mean(0)
            cls_pre.append(1 if a_predict.max() > 0.5 else 0)
            fpre_ = np.repeat(a_predict, 16)
            pl = len(fpre_)
            pre_dict[i] = fpre_
            gt_dict[i] = frame_gt[count: count + pl]
            count = count + pl
            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])
        # np.save('frame_label/xd_frame_pre_k_3.npy', frame_predict)
        # np.save('frame_label/xd_pre_dict_k_3.npy', pre_dict)
        # np.save('frame_label/xd_gt_dict_k_3.npy', gt_dict)
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print("auc:{}".format(auc_score))
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict, )
        ap_score = auc(recall, precision)

        print("accuracy:{}".format(accuracy))
        print("ap_score:{}".format(ap_score))


def get_xd_dataloader(args):
    normal_train_loader = torch.utils.data.DataLoader(
        XDVideo(root_dir=args.root_dir, mode='Train', modal=args.modal, num_segments=200,
                len_feature=args.len_feature, is_normal=True),
        batch_size=64,
        shuffle=True, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn, drop_last=True)
    abnormal_train_loader = torch.utils.data.DataLoader(
        XDVideo(root_dir=args.root_dir, mode='Train', modal=args.modal, num_segments=200,
                len_feature=args.len_feature, is_normal=False),
        batch_size=64,
        shuffle=True, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        XDVideo(root_dir=args.root_dir, mode='Test', modal=args.modal, num_segments=args.num_segments,
                len_feature=args.len_feature),
        batch_size=5,
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn)
    return normal_train_loader, abnormal_train_loader, test_loader


def get_xd_test_dataloader(args):
    test_loader = torch.utils.data.DataLoader(
        XDVideo(root_dir=args.root_dir, mode='Test', modal=args.modal, num_segments=args.num_segments,
                len_feature=args.len_feature),
        batch_size=5,
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn)
    return test_loader
