import torch
import numpy as np
from tqdm import tqdm
from dataloader import UCF_crime
from sklearn.metrics import roc_curve, auc, precision_recall_curve



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

        load_iter = iter(test_loader)
        frame_gt = np.load("Data/gt-ucf.npy")
        frame_predict = None
        ucf_pdict = {"Abuse": {},
                     "Arrest": {},
                     "Arson": {},
                     "Assault": {},
                     "Burglary": {},
                     "Explosion": {},
                     "Fighting": {},
                     "RoadAccidents": {},
                     "Robbery": {},
                     "Shooting": {},
                     "Shoplifting": {},
                     "Stealing": {},
                     "Vandalism": {},
                     "Normal": {},
                     }
        ucf_gdict = {"Abuse": {},
                     "Arrest": {},
                     "Arson": {},
                     "Assault": {},
                     "Burglary": {},
                     "Explosion": {},
                     "Fighting": {},
                     "RoadAccidents": {},
                     "Robbery": {},
                     "Shooting": {},
                     "Shoplifting": {},
                     "Stealing": {},
                     "Vandalism": {},
                     "Normal": {},
                     }
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        count = 0
        for i in tqdm(range(len(test_loader.dataset))):

            _data, _label, _name = next(load_iter)
            _name = _name[0]
            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                pl = len(a_predict) * 16

                if "Normal" in _name:
                    ucf_pdict["Normal"][_name] = np.repeat(a_predict, 16)
                    ucf_gdict["Normal"][_name] = frame_gt[count:count + pl]
                else:
                    ucf_pdict[_name[:-3]][_name] = np.repeat(a_predict, 16)
                    ucf_gdict[_name[:-3]][_name] = frame_gt[count:count + pl]
                count = count + pl
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros((0)).cuda()
        frame_gt = np.load("Data/gt-ucf.npy")
        # np.save('frame_label/ucf_pre_2.npy', frame_predict)
        # np.save('frame_label/ucf_pre_dict_2.npy', ucf_pdict)
        # np.save('frame_label/ucf_gt_dict_2.npy', ucf_gdict)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print(auc_score)
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict, )
        ap_score = auc(recall, precision)
        print(ap_score)

def test(net, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))
        load_iter = iter(test_loader)
        frame_gt = np.load("Data/gt-ucf.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)

            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()

                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros((0)).cuda()

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict, )
        ap_score = auc(recall, precision)

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)

def get_ucf_dataloader(args):
    normal_train_loader = torch.utils.data.DataLoader(
        UCF_crime(root_dir=args.root_dir, mode='Train', modal=args.modal, num_segments=200,
                  len_feature=args.len_feature, is_normal=True),
        batch_size=64,
        shuffle=True, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn, drop_last=True)
    abnormal_train_loader = torch.utils.data.DataLoader(
        UCF_crime(root_dir=args.root_dir, mode='Train', modal=args.modal, num_segments=200,
                  len_feature=args.len_feature, is_normal=False),
        batch_size=64,
        shuffle=True, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        UCF_crime(root_dir=args.root_dir, mode='Test', modal=args.modal, num_segments=args.num_segments,
                  len_feature=args.len_feature),
        batch_size=1,
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn)

    return normal_train_loader, abnormal_train_loader, test_loader

def get_ucf_test_dataloader(args):
    test_loader = torch.utils.data.DataLoader(
        UCF_crime(root_dir=args.root_dir, mode='Test', modal=args.modal, num_segments=args.num_segments,
                  len_feature=args.len_feature),
        batch_size=1,
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=args.worker_init_fn)

    return test_loader