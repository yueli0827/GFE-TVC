import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from triton.language import dtype

import wandb
import os

os.environ["WANDB_API_KEY"]='KEY'
os.environ["WANDB_MODE"]='offline'

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from video_dataset import VideoDataset
import tvc
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics

num_eps = 1e-11


def convert_list_to_ranges(data, output_file="segment_output.txt"):
    """
    将连续相同值的列表转换为范围表示，并同时打印到屏幕和写入txt文件。

    参数:
        data (list): 包含连续相同数值的列表
        output_file (str): 输出文件名，默认为"output.txt"
    """
    if not data:
        return []

    result = []
    start_index = 0
    current_value = data[0]

    for i in range(1, len(data)):
        if data[i] != current_value:
            # 当前值发生变化时，记录上一段的范围
            result.append((start_index, i - 1, current_value))
            start_index = i
            current_value = data[i]

    # 添加最后一段
    result.append((start_index, len(data) - 1, current_value))

    # 同时打印到屏幕和写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for start, end, value in result:
            line = f"{start} {end} {value}"
            print(line)  # 打印到屏幕
            f.write(line + '\n')  # 写入文件

    return result


class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha_train=0.3, alpha_eval=0.3,
                 n_ot_train=[50, 1], n_ot_eval=[50, 1], step_size=None, train_eps=0.06, eval_eps=0.01, ub_frames=False, ub_actions=True,
                 lambda_frames_train=0.05, lambda_actions_train=0.05, lambda_frames_eval=0.05, lambda_actions_eval=0.01,
                 temp=0.1, radius_gw=0.04, learn_clusters=True, n_frames=256, rho=0.1, exclude_cls=None, visualize=False):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.layer_sizes = layer_sizes
        self.exclude_cls = exclude_cls
        self.visualize = visualize

        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.step_size = step_size
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_frames = ub_frames
        self.ub_actions = ub_actions
        self.lambda_frames_train = lambda_frames_train
        self.lambda_actions_train = lambda_actions_train
        self.lambda_frames_eval = lambda_frames_eval
        self.lambda_actions_eval = lambda_actions_eval

        self.temp = temp
        self.n_frames = n_frames
        self.rho = rho

        # initialize cluster centers/codebook
        d = self.layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(self.n_clusters, d), dim=-1), requires_grad=learn_clusters)

        # initialize evaluation metrics
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.miou = ClusteringMetrics(metric='miou')
        self.save_hyperparameters()
        self.test_cache = []

    def training_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(features_raw.reshape(-1, features_raw.shape[-1]).reshape(B, T, D), dim=-1)
        codes = torch.exp(features @ self.clusters.T[None, ...] / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)
        with torch.no_grad():  # pseudo-labels from OT
            temp_prior = tvc.temporal_prior(T, self.n_clusters, self.rho, features.device)
            cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
            cost_matrix += temp_prior
            opt_codes, _ = tvc.segment_tvc(cost_matrix, mask, eps=self.train_eps, alpha=self.alpha_train,
                                            radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                            lambda_frames=self.lambda_frames_train,
                                            lambda_actions=self.lambda_actions_train, n_iters=self.n_ot_train,
                                            step_size=self.step_size)

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        self.log('train_loss', loss_ce)
        return loss_ce

    def validation_step(self, batch, batch_idx):  # subsample videos
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        # import pdb; pdb.set_trace()
        features = F.normalize(features_raw.reshape(-1, features_raw.shape[-1]).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        temp_prior = tvc.temporal_prior(T, self.n_clusters, self.rho, features.device)
        cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
        cost_matrix += temp_prior
        segmentation, _ = tvc.segment_tvc(cost_matrix, mask, eps=self.eval_eps, alpha=self.alpha_eval,
                                           radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                           lambda_frames=self.lambda_frames_eval,
                                           lambda_actions=self.lambda_actions_eval, n_iters=self.n_ot_eval,
                                           step_size=self.step_size)
        segments = segmentation.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per', metrics['mof'])
        self.log('val_f1_per', metrics['f1'])
        self.log('val_miou_per', metrics['miou'])

        # log validation loss
        codes = torch.exp(features @ self.clusters.T / self.temp)
        codes /= codes.sum(dim=-1, keepdim=True)
        pseudo_labels, _ = tvc.segment_tvc(cost_matrix, mask, eps=self.train_eps, alpha=self.alpha_train,
                                            radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                            lambda_frames=self.lambda_frames_train,
                                            lambda_actions=self.lambda_actions_train, n_iters=self.n_ot_train,
                                            step_size=self.step_size)
        loss_ce = -((pseudo_labels * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=[1, 2]).mean()
        self.log('val_loss', loss_ce)

        # plot qualitative examples of pseduo-labelling and embeddings for 5 videos evenly spaced in dataset
        # spacing =  int(self.trainer.num_val_batches[0] / 5)
        spacing =  int(self.trainer.num_val_batches[0] / 1)
        if batch_idx % spacing == 0 and wandb.run is not None and self.visualize:
            plot_idx = int(batch_idx / spacing)
            gt_cpu = gt[0].cpu().numpy()

            fdists = squareform(pdist(features[0].cpu().numpy(), 'cosine'))
            fig = plot_matrix(fdists, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(5, 5), xlabel='Frame index', ylabel='Frame index')
            wandb.log({f"val_pairwise_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()
            fig = plot_matrix(codes[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5), xlabel='Frame index', ylabel='Action index')
            wandb.log({f"val_P_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()
            fig = plot_matrix(pseudo_labels[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5), xlabel='Frame index', ylabel='Action index')
            wandb.log({f"val_OT_PL_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()
            fig = plot_matrix(segmentation[0].cpu().numpy().T, gt=gt_cpu, colorbar=False, title=fname[0], figsize=(10, 5), xlabel='Frame index', ylabel='Action index')
            wandb.log({f"val_OT_pred_{plot_idx}": fig, "trainer/global_step": self.trainer.global_step})
            plt.close()
        return None
    
    def test_step(self, batch, batch_idx):  # subsample videos
        features_raw, mask, gt, fname, n_subactions = batch
        D = self.layer_sizes[-1]
        B, T, _ = features_raw.shape
        features = F.normalize(features_raw.reshape(-1, features_raw.shape[-1]).reshape(B, T, D), dim=-1)

        # log clustering metrics over full epoch
        temp_prior = tvc.temporal_prior(T, self.n_clusters, self.rho, features.device)
        cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
        cost_matrix += temp_prior
        segmentation, _ = tvc.segment_tvc(cost_matrix, mask, eps=self.eval_eps, alpha=self.alpha_eval,
                                           radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                           lambda_frames=self.lambda_frames_eval,
                                           lambda_actions=self.lambda_actions_eval, n_iters=self.n_ot_eval,
                                           step_size=self.step_size)
        segments = segmentation.argmax(dim=2)
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('test_mof_per', metrics['mof'])
        self.log('test_f1_per', metrics['f1'])
        self.log('test_miou_per', metrics['miou'])

        # cache videos for plotting
        self.test_cache.append([metrics['mof'], segments, gt, mask, fname])

        return None
    
    def on_validation_epoch_end(self):
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('val_mof_full', mof)
        self.log('val_f1_full', f1)
        self.log('val_miou_full', miou)
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def on_test_epoch_end(self):
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        print('exclude_cls:', self.exclude_cls)
        print('pred_to_gt:', pred_to_gt)

        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _  = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('test_mof_full', mof)
        self.log('test_f1_full', f1)
        self.log('test_miou_full', miou)
        if wandb.run is not None and self.visualize:
            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                self.test_cache[i][0] = indep_eval_metrics(pred, gt, mask, ['mof'], exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)['mof']
            self.test_cache = sorted(self.test_cache, key=lambda x: x[0], reverse=True)

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache[:10]):
                pred_data = pred.cpu().numpy()
                print('mask:', mask)
                print('pred:', pred_data.shape, pred_data)

                # 转换并输出结果
                ranges = convert_list_to_ranges(pred_data.tolist()[0], 'segment_output.txt')
                print('pred_ranges:', ranges)

                fig = plot_segmentation_gt(gt, pred, mask, exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt,
                                           gt_uniq=np.unique(self.mof.gt_labels), name=f'{fname[0]}')
                wandb.log({f"test_segment_{i}": wandb.Image(fig), "trainer/global_step": self.trainer.global_step})
                plt.close()
        self.test_cache = []
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def fit_clusters(self, dataloader, K):
        cluster_centers_ = np.load('YOUR_CLUSTER_CENTERS_PATH')
        print('cluster_centers_:', cluster_centers_.dtype)
        print('cluster_centers_:', cluster_centers_.shape, cluster_centers_)
        self.clusters.data = torch.from_numpy(cluster_centers_.astype(np.float32)).to(self.clusters.device)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")

    # FUGW OT segmentation parameters
    parser.add_argument('--alpha-train', '-at', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--ub-frames', '-uf', action='store_true',
                        help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
    parser.add_argument('--ub-actions', '-ua', action='store_true',
                        help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
    parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05, help='penalty on balanced frames assumption for training')
    parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05, help='penalty on balanced actions assumption for training')
    parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05, help='penalty on balanced frames assumption for test')
    parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01, help='penalty on balanced actions assumption for test')
    parser.add_argument('--eps-train', '-et', type=float, default=0.07, help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.04, help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for tvc solver (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for tvc solver (eval)')
    parser.add_argument('--step-size', '-ss', type=float, default=None,
                        help='Step size/learning rate for tvc solver. Worth setting manually if ub-frames && ub-actions')

    # dataset params
    parser.add_argument('--base-path', '-p', type=str, default='/data/dataset/tvc/data', help='base directory for dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--activity', '-ac', type=str, nargs='+', required=True, help='activity classes to select for dataset')
    parser.add_argument('--exclude', '-x', type=int, default=None, help='classes to exclude from evaluation. use -1 for YTI')
    parser.add_argument('--n-frames', '-f', type=int, default=512, help='number of frames sampled per video for train/val')
    parser.add_argument('--n-frames-test', '-ft', type=int, default=512, help='number of frames sampled per video for test')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    
    # representation learning params
    parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--k-means', '-km', action='store_false', help='do not initialize clusters with kmeans default = True')
    parser.add_argument('--layers', '-ls', default=[64, 128, 40], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')
    parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
    parser.add_argument('--n-clusters', '-c', type=int, default=8, help='number of actions/clusters')

    # system/logging params
    parser.add_argument('--val-freq', '-vf', type=int, default=5, help='validation epoch frequency (epochs)')
    parser.add_argument('--gpu', '-g', type=int, default=1, help='gpu id to use')
    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb for logging')
    parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--eval', action='store_true', help='run evaluation on test set only')
    parser.add_argument('--group', type=str, default='base', help='wandb experiment group name')
    args = parser.parse_args()

    pl.seed_everything(args.seed)
        
    data_val = VideoDataset('/data/dataset/tvc/data', args.dataset, args.n_frames, standardise=args.std_feats, random=False, action_class=args.activity)
    data_train = VideoDataset('/data/dataset/tvc/data', args.dataset, args.n_frames, standardise=args.std_feats, random=True, action_class=args.activity)
    data_test = VideoDataset('/data/dataset/tvc/data', args.dataset, args.n_frames_test, standardise=args.std_feats, random=False, action_class=args.activity)
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4)

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    else:
        ssl = VideoSSL(layer_sizes=args.layers, n_clusters=args.n_clusters, alpha_train=args.alpha_train, alpha_eval=args.alpha_eval,
                       ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames_train=args.lambda_frames_train, lambda_frames_eval=args.lambda_frames_eval,
                       lambda_actions_train=args.lambda_actions_train, lambda_actions_eval=args.lambda_actions_eval, step_size=args.step_size,
                       train_eps=args.eps_train, eval_eps=args.eps_eval, radius_gw=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval,
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho, exclude_cls=args.exclude, visualize=args.visualize)
    activity_name = '_'.join(args.activity)
    name = f'{args.dataset}_{activity_name}_{args.group}_seed_{args.seed}'
    logger = pl.loggers.WandbLogger(name=name, project='video_ssl', save_dir='wandb') if args.wandb else None
    trainer = pl.Trainer(devices=[args.gpu], check_val_every_n_epoch=args.val_freq, max_epochs=args.n_epochs, log_every_n_steps=50, logger=logger)

    if args.k_means and args.ckpt is None:
        ssl.fit_clusters(test_loader, args.n_clusters)

    if not args.eval:
        trainer.validate(ssl, val_loader)
        trainer.fit(ssl, train_loader, val_loader)
    trainer.test(ssl, dataloaders=test_loader)
