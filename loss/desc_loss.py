import numpy as np
import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_fscore_support


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    """
    return torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """

    diffs = all_diffs(a, b)
    if metric == 'sqeuclidean':
        return torch.sum(diffs ** 2, dim=-1)
    elif metric == 'euclidean':
        return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
    elif metric == 'cityblock':
        return torch.sum(torch.abs(diffs), dim=-1)
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))
class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', safe_radius=0.10):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.metric = metric
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts, dist=torch.Tensor([])):
        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        if dist.nelement() == 0:
            dist = cdist(anchor, positive, metric=self.metric)
        dist_keypts = np.eye(dist_keypts.shape[0]) * 10 + dist_keypts.detach().cpu().numpy()
        add_matrix = torch.zeros_like(dist)
        add_matrix[np.where(dist_keypts < self.safe_radius)] += 10
        dist = dist + add_matrix
        return self.calculate_loss(dist, pids)

    def calculate_loss(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.

        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """
        # generate the mask that mask[i, j] reprensent whether i th and j th are from the same identity.
        # torch.equal is to check whether two tensors have the same size and elements
        # torch.eq is to computes element-wise equality
        same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        # negative_mask = np.logical_not(same_identity_mask)

        # dists * same_identity_mask get the distance of each valid anchor-positive pair.
        furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
        # here we use "dists +  10000*same_identity_mask" to avoid the anchor-positive pair been selected.
        closest_negative, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=1)
        # closest_negative_row, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=0)
        # closest_negative = torch.min(closest_negative_col, closest_negative_row)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]
        loss = torch.max(furthest_positive - self.pos_margin, torch.zeros_like(diff)) + torch.max(
            self.neg_margin - closest_negative, torch.zeros_like(diff))

        return torch.mean(loss), closest_negative / (furthest_positive+1e-6), accuracy #

class ContrastiveLossWithSOS(ContrastiveLoss):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', safe_radius=0.10, sos_weight=0.1):
        """
        Contrastive Loss with Second-Order Similarity Regularization (SOS).
        Inherits from ContrastiveLoss.

        Args:
            pos_margin (float): Margin for positive pairs.
            neg_margin (float): Margin for negative pairs.
            metric (str): Distance metric ('euclidean', 'sqeuclidean', or 'cityblock').
            safe_radius (float): Radius to filter keypoints.
            sos_weight (float): Weight for the SOS Regularization term.
        """
        super(ContrastiveLossWithSOS, self).__init__(pos_margin, neg_margin, metric, safe_radius)
        self.sos_weight = sos_weight

    def sos_regularization(self, anchor, positive):
        """
        Compute Second-Order Similarity Regularization Loss.

        Args:
            anchor (torch.Tensor): Anchor descriptors of shape (B, C).
            positive (torch.Tensor): Positive descriptors of shape (B, C).

        Returns:
            torch.Tensor: SOS Regularization loss.
        """
        # Pairwise distances within anchor and positive
        d_x = torch.cdist(anchor, anchor, p=2)  # d(x_i, x_j)
        d_xp = torch.cdist(positive, positive, p=2)  # d(x_i^+, x_j^+)

        # Exclude diagonal elements (j ≠ i)
        mask = ~torch.eye(d_x.size(0), dtype=torch.bool, device=anchor.device)  # Mask for j ≠ i

        # Compute Second-Order Distance
        second_order_diff = (d_x - d_xp) ** 2
        second_order_loss = torch.sqrt(torch.sum(second_order_diff * mask, dim=1))  # Sum over j ≠ i, take sqrt

        # Average over all anchor-positive pairs
        sos_reg = torch.mean(second_order_loss)
        return sos_reg

    def forward(self, anchor, positive, dist_keypts, dist=torch.Tensor([])):
        """
        Compute the combined Contrastive Loss and SOS Regularization Loss.

        Args:
            anchor (torch.Tensor): Anchor descriptors of shape (B, C).
            positive (torch.Tensor): Positive descriptors of shape (B, C).
            dist_keypts (torch.Tensor): Distance matrix for keypoints of shape (B, B).
            dist (torch.Tensor): Precomputed distance matrix (optional).

        Returns:
            torch.Tensor: Total loss (Contrastive Loss + SOS Regularization).
            torch.Tensor: Contrastive Loss.
            torch.Tensor: SOS Regularization Loss.
        """
        # Contrastive Loss (inherited from parent class)
        contrastive_loss, diff, accuracy = super().forward(anchor, positive, dist_keypts, dist)

        # Second-Order Similarity Loss
        sos_loss = self.sos_regularization(anchor, positive)

        # Combine Losses
        total_loss = contrastive_loss + self.sos_weight * sos_loss

        return total_loss, diff, accuracy

class Hardest_ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', safe_radius=0.10):
        super(Hardest_ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.metric = metric
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts):
        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        dist = cdist(anchor, positive, metric=self.metric)
        dist_keypts = np.eye(dist_keypts.shape[0]) * 10 + dist_keypts.detach().cpu().numpy()
        add_matrix = torch.zeros_like(dist)
        add_matrix[np.where(dist_keypts < self.safe_radius)] += 10
        dist = dist + add_matrix
        return self.calculate_loss(dist, pids)

    def calculate_loss(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.

        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """
        same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))

        # dists * same_identity_mask get the distance of each valid anchor-positive pair.
        furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
        # here we use "dists +  10000*same_identity_mask" to avoid the anchor-positive pair been selected.
        closest_neg_col = torch.min(dists + 1e5 * same_identity_mask, dim=1)
        closest_neg_row = torch.min(dists + 1e5 * same_identity_mask, dim=0)
        closest_negative = torch.min(closest_neg_col[0], closest_neg_row[0])
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]

        loss = F.relu(furthest_positive.pow(2) - self.pos_margin) + \
               0.5 * F.relu(self.neg_margin - closest_neg_col[0]).pow(2) + \
               0.5 * F.relu(self.neg_margin - closest_neg_row[0]).pow(2)

        return torch.mean(loss), accuracy


class ClassificationLoss(nn.Module):
    """
    Classification loss class. Creates a ClassificationLoss object that is used to supervise the inlier/outlier classification of the putative correspondences.

    """

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.w_class = 1
        self.compute_stats = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def class_loss(self, predicted, target):
        """
        Binary classification loss per putative correspondence.

        Args:
            predicted (torch tensor): predicted weight per correspondence [b,n,1]
            target (torch tensor): ground truth label per correspondence (0 - outlier, 1 - inlier) [b,n,1]

        Return:
            class_loss (torch tensor): binary cross entropy loss [b]
        """

        loss = nn.BCELoss(
            reduction='none')  # Binary Cross Entropy loss, expects that the input was passed through the sigmoid
        sigmoid = nn.Sigmoid()

        predicted_labels = sigmoid(predicted).flatten().to(self.device)

        class_loss = loss(predicted_labels, target.flatten()).reshape(predicted.shape[0], -1)

        # Computing weights for compensating the class imbalance

        is_pos = (target.squeeze(-1) < 0.5).type(target.type())
        is_neg = (target.squeeze(-1) > 0.5).type(target.type())

        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
        class_loss_p = torch.sum(class_loss * is_pos, dim=1)
        class_loss_n = torch.sum(class_loss * is_neg, dim=1)
        class_loss = class_loss_p * 0.5 / num_pos + class_loss_n * 0.5 / num_neg

        return class_loss

    def forward(self, predicted, target, scores=None):
        """
        Evaluates the binary cross entropy classification loss

        Args:
            predicted (torch tensor): predicted logits per correspondence [b,n]
            target (torch tensor): ground truth label per correspondence (0 - outlier, 1 - inlier) [b,n,1]
            scores (torch tensor): predicted score (weight) per correspondence (0 - outlier, 1 - inlier) [b,n]

        Return:
            loss (torch tensor): mean binary cross entropy loss
            precision (numpy array): Mean classification precision (inliers)
            recall (numpy array): Mean classification recall (inliers)
        """
        predicted = predicted.to(self.device)
        target = target.to(self.device)

        class_loss = self.class_loss(predicted, target)

        loss = torch.tensor([0.]).to(self.device)

        if self.w_class > 0:
            loss += torch.mean(self.w_class * class_loss)

        if self.compute_stats:
            assert scores != None, "If precision and recall should be computed, scores cannot be None!"

            y_predicted = scores.detach().cpu().numpy().reshape(-1)
            y_gt = target.detach().cpu().numpy().reshape(-1)

            precision, recall, f_measure, _ = precision_recall_fscore_support(y_gt, y_predicted.round(),
                                                                              average='binary')

            return loss, precision, recall

        else:
            return loss, None, None


class TransformationLoss(nn.Module):
    """
    Transformation loss class. Creates a TransformationLoss object that is used to supervise the rotation and translation estimation part of the network.

    Args:
        cfg (dict): configuration parameters

    """

    def __init__(self):
        super(TransformationLoss, self).__init__()
        self.trans_loss_type = 3
        self.trans_loss_iter = 15000
        self.w_trans = 0.4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trans_loss_margin = 0.1
        self.inlier_threshold = 0.075

    def trans_loss(self, x_in, rot_est, trans_est, gt_rot_mat, gt_t_vec):
        """
        Loss function on the transformation parameter. Based on the selected type of the loss computes either:
        0 - Vector distance between the point reconstructed using the EST transformation paramaters and the putative correspondence
        1 - Frobenius norm on the rotation matrix and L2 norm on the translation vector
        2 - L2 distance between the points reconstructed using the estimated and GT transformation paramaters
        3 - L1 distance between the points reconstructed using the estimated and GT transformation paramaters

        Args:
            x_in (torch tensor): coordinates of the input point [b,1,n,6]
            rot_est (torch tensor): currently estimated rotation matrices [b,3,3]
            trans_est (torch tensor): currently estimated translation vectors [b,3,1]
            gt_rot_mat (torch tensor): ground truth rotation matrices [b,3,3]
            gt_t_vec (torch tensor): ground truth translation vectors [b,3,1]

        Return:
            r_loss (torch tensor): transformation loss if type 0 or 2 else Frobenius norm of the rotation matrices [b,1]
            t_loss (torch tensor): 0 if type 0, 2 or 3 else L2 norm of the translation vectors [b,1]
        """
        if self.trans_loss_type == 0:

            x2_reconstruct = torch.matmul(rot_est, x_in[:, 0, :, 0:3].transpose(1, 2)) + trans_est
            r_loss = torch.mean(
                torch.mean(torch.norm(x2_reconstruct.transpose(1, 2) - x_in[:, :, :, 3:6], dim=(1)), dim=1))
            t_loss = torch.zeros_like(r_loss)

        elif self.trans_loss_type == 1:
            r_loss = torch.norm(gt_rot_mat - rot_est, dim=(1, 2))
            t_loss = torch.norm(trans_est - gt_t_vec, dim=1)  # Torch norm already does sqrt (p=1 for no sqrt)

        elif self.trans_loss_type == 2:
            x2_reconstruct_estimated = torch.matmul(rot_est, x_in[:, 0, :, 0:3].transpose(1, 2)) + trans_est
            x2_reconstruct_gt = torch.matmul(gt_rot_mat, x_in[:, 0, :, 0:3].transpose(1, 2)) + gt_t_vec

            r_loss = torch.mean(torch.norm(x2_reconstruct_estimated - x2_reconstruct_gt, dim=1), dim=1)
            t_loss = torch.zeros_like(r_loss)

        elif self.trans_loss_type == 3:
            x2_reconstruct_estimated = torch.matmul(rot_est, x_in[:, 0, :, 0:3].transpose(1, 2)) + trans_est
            x2_reconstruct_gt = torch.matmul(gt_rot_mat, x_in[:, 0, :, 0:3].transpose(1, 2)) + gt_t_vec

            r_loss = torch.mean(torch.sum(torch.abs(x2_reconstruct_estimated - x2_reconstruct_gt), dim=1), dim=1)
            t_loss = torch.zeros_like(r_loss)

        return r_loss, t_loss

    def forward(self, global_step, data, rot_est, trans_est):
        """
        Evaluates the pairwise loss function based on the current values

        Args:
            global_step (int): current training iteration (used for controling which parts of the loss are used in the current iter) [1]
            data (dict): input data of the current batch
            rot_est (torch tensor): rotation matrices estimated based on the current scores [b,3,3]
            trans_est  (torch tensor): translation vectors estimated based on the current scores [b,3,1]

        Return:
            loss (torch tensor): mean transformation loss of the current iteration over the batch
            loss_raw (torch tensor): mean transformation loss of the current iteration (return value for tenbsorboard before the trans loss is plugged in )
        """

        # Extract the current data
        x_in, gt_R, gt_t = data['xs'].to(self.device), data['R'].to(self.device), data['t'].to(self.device)
        gt_inlier_ratio = data['inlier_ratio'].to(self.device)

        # Compute the transformation loss
        r_loss, t_loss = self.trans_loss(x_in, rot_est, trans_est, gt_R, gt_t)

        # Extract indices of pairs with a minimum inlier ratio (do not propagate Transformation loss if point clouds do not overlap)
        idx_inlier_ratio = gt_inlier_ratio > self.inlier_threshold
        inlier_ratio_mask = torch.zeros_like(r_loss)
        inlier_ratio_mask[idx_inlier_ratio] = 1

        loss_raw = torch.tensor([0.]).to(self.device)

        if self.w_trans > 0:
            r_loss *= inlier_ratio_mask
            t_loss *= inlier_ratio_mask

            loss_raw += torch.mean(
                torch.min(self.w_trans * (r_loss + t_loss), self.trans_loss_margin * torch.ones_like(t_loss)))

        # Check global_step and add essential loss
        loss = loss_raw if global_step >= self.trans_loss_iter else torch.tensor([0.]).to(self.device)

        return loss, loss_raw
