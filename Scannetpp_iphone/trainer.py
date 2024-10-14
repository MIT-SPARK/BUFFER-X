import gc
from utils.SE3 import *
from utils.timer import Timer, AverageMeter
from loss.desc_loss import ContrastiveLoss, cdist
from tensorboardX import SummaryWriter
from utils.common import make_open3d_point_cloud, ensure_dir


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.cfg = args.cfg
        self.train_modal = self.cfg.stage
        self.epoch = self.cfg.train.epoch
        self.save_dir = args.save_dir

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.evaluate_interval = args.evaluate_interval
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        self.desc_loss = ContrastiveLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.L1_loss = torch.nn.L1Loss()

        # create meters and timers
        self.meter_list = ['ref_loss', 'ref_error', 'desc_loss', 'desc_acc', 'eqv_loss', 'eqv_acc', 'det_loss',
                           'match_loss']
        self.meter_dict = {}
        for key in self.meter_list:
            self.meter_dict[key] = AverageMeter()

    def get_matching_indices(self, source, target, relt_pose, search_voxel_size):
        """
        Input
            - source:     [N, 3]
            - target:     [M, 3]
            - relt_pose:  [4, 4]
        Output:
            - match_inds: [C, 2]
        """
        source = transform(source, relt_pose)
        diffs = source[:, None] - target[None]
        dist = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
        min_ind = torch.cat([torch.arange(source.shape[0])[:, None].cuda(), torch.argmin(dist, dim=1)[:, None]], dim=-1)
        min_val = torch.min(dist, dim=1)[0]
        match_inds = min_ind[min_val < search_voxel_size]

        return match_inds

    def train(self):
        best_loss = 1000000000

        for epoch in range(self.epoch):
            gc.collect()
            self.train_epoch(epoch)
            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate()
                print(f'Evaluation: Epoch {epoch}')
                for key in res.keys():
                    print(f"{key}: {res[key]}")
                    self.writer.add_scalar(key, res[key], epoch)

                if self.train_modal == 'Ref':
                    if res['ref_loss'] < best_loss:
                        best_loss = res['ref_loss']
                        self._snapshot('best')
                elif self.train_modal == 'Desc':
                    if res['desc_loss'] < best_loss:
                        best_loss = res['desc_loss']
                        self._snapshot('best')
                elif self.train_modal == 'Keypt':
                    if res['det_loss'] < best_loss:
                        best_loss = res['det_loss']
                        self._snapshot('best')
                elif self.train_modal == 'Inlier':
                    if res['match_loss'] < best_loss:
                        best_loss = res['match_loss']
                        self._snapshot('best')
                else:
                    raise NotImplementedError

            if (epoch + 1) % self.scheduler_interval == 0:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                print('update detector learning rate: %f -> %f' % (old_lr, new_lr))

            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

        # finish all epoch
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        print('training start!!')
        self.model.train()
        data_timer, model_timer = Timer(), Timer()

        num_batch = len(self.train_loader)
        num_iter = min(self.cfg.train.max_iter, num_batch)
        data_iter = iter(self.train_loader)
        for i in range(num_iter):
            data_timer.tic()
            data_source = data_iter.__next__()

            # compute normals
            src_pts, tgt_pts = data_source['src_pcd'], data_source['tgt_pcd']
            src_pcd = make_open3d_point_cloud(src_pts.numpy(), [1, 0.706, 0])
            src_pcd.estimate_normals()
            src_pcd.orient_normals_towards_camera_location()
            src_normls = np.array(src_pcd.normals)
            tgt_pcd = make_open3d_point_cloud(tgt_pts.numpy(), [0, 0.651, 0.929])
            tgt_pcd.estimate_normals()
            tgt_pcd.orient_normals_towards_camera_location()
            tgt_normls = np.array(tgt_pcd.normals)
            data_source['features'] = torch.from_numpy(np.concatenate([src_normls, tgt_normls], axis=0)).float()

            data_timer.toc()
            model_timer.tic()
            # forward
            self.optimizer.zero_grad()
            output = self.model(data_source)
            if output is None:
                continue

            if self.train_modal == 'Ref':

                # angle errors between src and tgt
                src_axis, tgt_axis = output['src_ref'], output['tgt_ref']
                gt_trans = data_source['relt_pose'].to(src_axis.device)
                src_axis = src_axis @ gt_trans[:3, :3].transpose(-1, -2)
                err = 1 - torch.cosine_similarity(src_axis, tgt_axis).abs()

                # probabilistic cosine loss
                src_s, tgt_s = output['src_s'], output['tgt_s']
                eps = (src_s[:, 0] + tgt_s[:, 0]) / 2
                ref_loss = (torch.log(eps) + err / eps).mean()

                loss = ref_loss
                if torch.isnan(loss):
                    print(f"{data_source['src_id']} {data_source['tgt_id']} loss is nan, skip")
                    continue
                stats = {
                    "ref_loss": float(loss.item()),
                    'ref_error': float(err.mean().item())
                }

            if self.train_modal == 'Desc':
                # descriptor loss
                tgt_kpt, src_des, tgt_des = output['tgt_kpt'], output['src_des'], output['tgt_des']
                desc_loss, diff, accuracy = self.desc_loss(src_des, tgt_des, cdist(tgt_kpt, tgt_kpt))

                # equivariant loss to make two cylindrical maps similar
                eqv_loss = self.class_loss(output['equi_score'], output['gt_label'])
                pre_label = torch.argmax(output['equi_score'], dim=1)
                eqv_acc = (pre_label == output['gt_label']).sum() / pre_label.shape[0]

                # refer to RoReg(https://github.com/HpWang-whu/RoReg)
                loss = 4 * desc_loss + eqv_loss
                if torch.isnan(loss):
                    print(f"{data_source['src_id']} {data_source['tgt_id']} loss is nan, skip")
                    continue
                stats = {
                    "desc_loss": float(desc_loss.item()),
                    "desc_acc": float(accuracy.item()),
                    "eqv_loss": float(eqv_loss.item()),
                    "eqv_acc": float(eqv_acc.item()),
                }

            if self.train_modal == 'Keypt':

                src_s, tgt_s = output['src_s'], output['tgt_s']
                src_kpt, src_des, tgt_des = output['src_kpt'], output['src_des'], output['tgt_des']
                desc_loss, diff, accuracy = self.desc_loss(src_des, tgt_des, cdist(src_kpt, src_kpt))

                # det loss
                sigma = (src_s[:, 0] + tgt_s[:, 0]) / 2
                det_loss = torch.mean((1.05 - diff.detach()) * sigma)

                loss = det_loss
                if torch.isnan(loss):
                    print(f"{data_source['src_id']} {data_source['tgt_id']} loss is nan, skip")
                    continue
                stats = {
                    'det_loss': float(det_loss.item()),
                    "desc_acc": float(accuracy.item()),
                }

            if self.train_modal == 'Inlier':

                # L1 loss
                pred_ind, gt_ind = output['pred_ind'], output['gt_ind']
                match_loss = self.L1_loss(pred_ind, gt_ind)

                loss = match_loss
                if torch.isnan(loss):
                    print(f"{data_source['src_id']} {data_source['tgt_id']} loss is nan, skip")
                    continue
                stats = {
                    "match_loss": float(match_loss.item()),
                }

            # backward
            loss.backward()
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                self.optimizer.step()
            model_timer.toc()
            torch.cuda.empty_cache()

            for key in self.meter_list:
                if stats.get(key) is not None:
                    self.meter_dict[key].update(stats[key])
               
            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1} [{i + 1:4d}/{num_iter}] "
                      f"data_time: {data_timer.avg:.2f}s "
                      f"model_time: {model_timer.avg:.2f}s ")
                for key in self.meter_dict.keys():
                    print(f"{key}: {self.meter_dict[key].avg:.6f}")
                    self.meter_dict[key].reset()
        self._snapshot(f'{epoch}')

    def evaluate(self):
        print('validation start!!')
        self.model.eval()
        data_timer, model_timer = Timer(), Timer()

        with torch.no_grad():
            num_batch = len(self.val_loader)
            data_iter = iter(self.val_loader)
            for i in range(num_batch):
                data_timer.tic()
                data_source = data_iter.__next__()

                # compute normals
                src_pts, tgt_pts = data_source['src_pcd'], data_source['tgt_pcd']
                src_pcd = make_open3d_point_cloud(src_pts.numpy(), [0.2, 0.3, 0.4])
                src_pcd.estimate_normals()
                src_pcd.orient_normals_towards_camera_location()
                src_normls = np.array(src_pcd.normals)
                tgt_pcd = make_open3d_point_cloud(tgt_pts.numpy(), [0.7, 0.8, 0.9])
                tgt_pcd.estimate_normals()
                tgt_pcd.orient_normals_towards_camera_location()
                tgt_normls = np.array(tgt_pcd.normals)
                data_source['features'] = torch.from_numpy(np.concatenate([src_normls, tgt_normls], axis=0)).float()

                data_timer.toc()
                model_timer.tic()
                # forward
                output = self.model(data_source)
                if output is None:
                    continue

                if self.train_modal == 'Ref':
                    src_axis, tgt_axis = output['src_ref'], output['tgt_ref']
                    gt_trans = data_source['relt_pose'].to(src_axis.device)
                    src_axis = src_axis @ gt_trans[:3, :3].transpose(-1, -2)
                    err = 1 - torch.cosine_similarity(src_axis, tgt_axis).abs()

                    src_s, tgt_s = output['src_s'], output['tgt_s']
                    eps = (src_s[:, 0] + tgt_s[:, 0]) / 2
                    ref_loss = (torch.log(eps) + err / eps).mean()

                    stats = {
                        "ref_loss": float(ref_loss.item()),
                        'ref_error': float(err.mean().item())
                    }

                if self.train_modal == 'Desc':
                    src_kpt, src_des, tgt_des = output['src_kpt'], output['src_des'], output['tgt_des']
                    desc_loss, diff, accuracy = self.desc_loss(src_des, tgt_des, cdist(src_kpt, src_kpt))
                    eqv_loss = self.class_loss(output['equi_score'], output['gt_label'])
                    pre_label = torch.argmax(output['equi_score'], dim=1)
                    eqv_acc = (pre_label == output['gt_label']).sum() / pre_label.shape[0]

                    stats = {
                        "desc_loss": float(desc_loss.item()),
                        "desc_acc": float(accuracy.item()),
                        "eqv_loss": float(eqv_loss.item()),
                        "eqv_acc": float(eqv_acc.item()),
                    }

                if self.train_modal == 'Keypt':
                    src_s, tgt_s = output['src_s'], output['tgt_s']
                    src_kpt, src_des, tgt_des = output['src_kpt'], output['src_des'], output['tgt_des']
                    desc_loss, diff, accuracy = self.desc_loss(src_des, tgt_des, cdist(src_kpt, src_kpt))

                    sigma = (src_s[:, 0] + tgt_s[:, 0]) / 2
                    det_loss = torch.mean((1.05 - diff.detach()) * sigma)

                    loss = det_loss
                    stats = {
                        'det_loss': float(det_loss.item()),
                        "desc_acc": float(accuracy.item()),
                    }

                if self.train_modal == 'Inlier':
                    pred_ind, gt_ind = output['pred_ind'], output['gt_ind']
                    match_loss = self.L1_loss(pred_ind, gt_ind)

                    stats = {
                        "match_loss": float(match_loss.item()),
                    }

                model_timer.toc()
                torch.cuda.empty_cache()
                for key in self.meter_list:
                    if stats.get(key) is not None:
                        self.meter_dict[key].update(stats[key])

        self.model.train()
        res = {}
        for key in self.meter_dict.keys():
            res[key] = self.meter_dict[key].avg

        return res

    def _snapshot(self, info):
        save_path = self.cfg.snapshot_root + f'/{self.train_modal}'
        ensure_dir(save_path)
        torch.save(self.model.module.state_dict(), save_path + f'/{info}.pth')
        print(f"Save model to {save_path}/{info}.pth")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
