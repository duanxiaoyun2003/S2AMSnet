import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from packaging import version

class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
    
    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        elif self.reduction == 'mean':
            return torch.mean(loss)
        
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class PatchNCELoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, batch_size):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device = torch.device("cuda")):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):  #掩码相关样本 (mask_correlated_samples):
        N = 2 * batch_size  #结合两个批次后的样本总数。
        mask = torch.ones((N, N))  #一个 (N, N) 的矩阵，初始化为 1，用于排除特定样本对。
        mask = mask.fill_diagonal_(0)  #将掩码矩阵的对角线设置为 0，以排除自相似度。
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            #循环: 设置特定的非对角元素为 0，确保正样本对（来自不同视图的相同样本）不被用作负样本对。
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        # z_i = F.normalize(z_i, dim=0)
        # z_j = F.normalize(z_j, dim=0)
        z_i = z_i.view(self.batch_size, -1)
        z_j = z_j.view(self.batch_size, -1)
        # z_i = F.normalize(z_i, dim=1)
        # z_j = F.normalize(z_j, dim=1)

        z = torch.cat((z_i, z_j), dim=0)  #从源域图像SD和生成域GD中得到的嵌入。它们被重塑并拼接。

        sim = torch.matmul(z, z.T) / self.temperature  #计算嵌入之间的相似度矩阵，并用温度缩放。
        sim_i_j = torch.diag(sim, self.batch_size)  #z_i 和 z_j 之间的相似度（正样本对）。
        sim_j_i = torch.diag(sim, -self.batch_size)  #z_j 和 z_i 之间的相似度（从另一角度看的正样本对）。

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  #组合正样本对的相似度。
        negative_samples = sim[self.mask].reshape(N, -1)  #使用掩码提取所有负样本对的相似度。

        labels = torch.zeros(N).to(positive_samples.device).long()  #交叉熵损失的目标标签，正样本对标记为 0。
        logits = torch.cat((positive_samples, negative_samples), dim=1) #将正样本和负样本组合成矩阵。
        loss = self.criterion(logits, labels)
        loss /= N #计算损失，取平均。

        return loss
        #InstanceLoss 类定义了一种对比损失函数，旨在使相同实例的不同视图在嵌入空间中接近，
        #而使不同实例的嵌入距离远离。它使用温度参数来调整相似度分数，并应用交叉熵损失以确保正确的嵌入分离。
    

# def MI_loss(src, tgt, label, nets, args):
def MI_loss(src, tgt, nets, args):
    #用于计算对比学习中的 Mutual Information (MI) 互信息损失。
    #它的主要目的是通过计算不同视图或增强下的特征之间的对比损失来训练模型。
    nce_layers = args.nce_layers
    nce_layers = list(map(int, nce_layers.split(',')))
    #从 args 中获取 nce_layers 参数，并将其转换为整数列表。

    bs = src.shape[0] #计算批量大小，即输入数据的第一个维度。

    crit  = InstanceLoss(batch_size=bs, temperature=args.temp).cuda()
    #创建 InstanceLoss 实例损失 实例，设置批量大小和温度，并将其移动到 GPU。
    # crit = SupConLoss(temperature=args.temp).cuda()
    netG = nets
    # netG, netMLP = nets

    n_layers = len(nce_layers)
    feat_q = netG(tgt, nce_layers, encode_only=True)

    feat_k = netG(src, nce_layers, encode_only=True)
    #使用网络 netG 计算目标数据 tgt 和原始数据 src 的特征。这些特征是在指定的层中计算的。

    # feat_q = netMLP(feat_q)
    # feat_k = netMLP(feat_k)
    # feat_k_pool, sample_ids = netF(feat_k, 256, None)
    # feat_q_pool, _ = netF(feat_q, 256, sample_ids)

    # bs = src.shape[0]

    total_nce_loss = 0.0
    for f_q, f_k, nce_layer in zip(feat_q, feat_k, nce_layers):
        # f_q = torch.reshape(f_q, (bs, -1))
        # f_q = F.normalize(f_q, dim=1)
        # f_k = torch.reshape(f_k, (bs, -1))
        # f_k = F.normalize(f_k, dim=1)
        # z = torch.cat([f_q.unsqueeze(1), f_k.unsqueeze(1)], dim=1)
        # loss = crit(z, label) * 1.0
        loss = crit(f_q, f_k) * 1.0
        total_nce_loss += loss.mean()
        #遍历 feat_q 和 feat_k 中的每一对特征，以及对应的层。
        #计算每一对特征的损失，并累加到 total_nce_loss 中。


    return total_nce_loss / n_layers
    #将总损失除以层数，返回平均损失值。