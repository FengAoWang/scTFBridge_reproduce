import torch
import torch.nn as nn
import torch.nn.functional as F
import scvi
import scanpy as sc
import anndata
from statsmodels.imputation.bayes_mi import MI

from utils.loss_function import KL_loss, Reconstruction_loss
from model.CLUB import MIEstimator

def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std).cuda()
    return epsilon * std + mean


def product_of_experts(mu_set_, log_var_set_):
    tmp = 0
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]))

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(poe_var)

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i])) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var


class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, mask, latent_dim, bias=False):
        # mask 应该和转置后的权重维度相同
        # n_input x n_output_nodes
        if latent_dim != mask.shape[0] or n_out != mask.shape[1]:
            raise ValueError('Incorrect shape of the mask.')

        super().__init__(n_in, n_out, bias)

        self.register_buffer('mask', mask.t())
        self.latent_dim = latent_dim

        # 初始化权重时，只保留 mask 不为 0 的部分
        with torch.no_grad():
            self.weight[:, :self.latent_dim] *= self.mask  # 仅初始化非零的 mask 部分
            # torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # 克隆权重来避免原地操作
        masked_weight = self.weight.clone()

        # 仅保留 mask 不为 0 的地方的权重值
        masked_weight[:, :self.latent_dim] = masked_weight[:, :self.latent_dim] * self.mask + (1 - self.mask) * self.weight[:, :self.latent_dim].detach()

        # 计算前向传播
        x = nn.functional.linear(x, masked_weight)
        return x


class LinearLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 batchnorm: bool = False,
                 activation=None,
                 mask=None):
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None

        self.activation = None
        self.mask = mask
        if activation is not None:
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'sigmoid':
                self.activation = torch.sigmoid
            elif activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'leakyrelu':
                self.activation = torch.nn.LeakyReLU()
            elif activation == 'selu':
                self.activation = torch.nn.SELU()

    def forward(self, input_x):
        # 如果 mask 存在，将它应用到线性层的权重上
        if self.mask is not None:
            masked_weight = self.linear_layer.weight * self.mask.T
            x = F.linear(input_x, masked_weight, self.linear_layer.bias)
        else:
            x = self.linear_layer(input_x)

        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ModalVAEEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 latent_dim,
                 activation=None):
        super(ModalVAEEncoder, self).__init__()
        self.FeatureEncoder = nn.ModuleList([LinearLayer(input_dim, hidden_dims[0],
                                                         batchnorm=True, activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureEncoder.append(LinearLayer(hidden_dims[i], hidden_dims[i+1],
                                                   batchnorm=True, activation=activation))

        self.mu_predictor = LinearLayer(hidden_dims[-1], latent_dim, batchnorm=True)
        self.logVar_predictor = LinearLayer(hidden_dims[-1], latent_dim, batchnorm=True)

    def forward(self, input_x):
        for layer in self.FeatureEncoder:
            input_x = layer(input_x)
        mu = self.mu_predictor(input_x)
        log_var = self.logVar_predictor(input_x)
        latent_z = reparameterize(mu, log_var)
        return mu, log_var, latent_z

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 activation=None,
                 mask=None):
        super(Encoder, self).__init__()
        if mask is not None:
            self.FeatureEncoder = nn.ModuleList(
                [MaskedLinear(input_dim, hidden_dims[0], mask)])
        else:
            self.FeatureEncoder = nn.ModuleList(
                [LinearLayer(input_dim, hidden_dims[0], batchnorm=True, activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureEncoder.append(
                LinearLayer(hidden_dims[i], hidden_dims[i + 1], batchnorm=True, activation=activation))

    def forward(self, input_x):
        for layer in self.FeatureEncoder:
            input_x = layer(input_x)
        return input_x


class ModalVAEDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation=None,
                 ):
        super(ModalVAEDecoder, self).__init__()
        self.FeatureDecoder = nn.ModuleList([LinearLayer(input_dim, hidden_dims[0],
                                                         dropout=0.2, batchnorm=True,
                                                         activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureDecoder.append(LinearLayer(hidden_dims[i], hidden_dims[i+1],
                                                   dropout=0.2, batchnorm=True,
                                                   activation=activation))
        self.ReconsPredictor = LinearLayer(hidden_dims[-1], output_dim)

    def forward(self, input_x):
        for layer in self.FeatureDecoder:
            input_x = layer(input_x)
        DataRecons = self.ReconsPredictor(input_x)
        return DataRecons


class scMulti(nn.Module):
    def __init__(self,
                 input_dims,
                 encoder_hidden_dims,
                 decoder_hidden_dims,
                 latent_dim,
                 kl_weight,
                 dist,
                 batch_dims,
                 con_weight,
                 mask):
        super(scMulti, self).__init__()
        self.kl_weight = kl_weight
        self.con_weight = con_weight
        self.temperature = 1
        self.input_dims = input_dims
        self.batch_dims = batch_dims

        self.RNADataDist = dist[0]
        self.ATACDataDist = dist[1]

        #   modality private Encoder
        self.RNAEncoder = Encoder(input_dims[0], encoder_hidden_dims, activation='relu')
        self.ATACEncoder = Encoder(input_dims[1], encoder_hidden_dims, activation='relu')

        self.RNAPrivateEncoder = ModalVAEEncoder(encoder_hidden_dims[-1], [256], latent_dim, activation='relu')
        self.ATACPrivateEncoder = ModalVAEEncoder(encoder_hidden_dims[-1], [256], latent_dim, activation='relu')

        #   modality share Encoder
        self.ModalShareEncoder = ModalVAEEncoder(encoder_hidden_dims[-1], [256], latent_dim, activation='relu')

        #   modality private Decoder
        self.RNADecoder = ModalVAEDecoder(latent_dim * 2 + batch_dims, decoder_hidden_dims, input_dims[0], activation='relu')

        self.ATACDecoder = MaskedLinear(latent_dim * 2 + batch_dims, input_dims[1], mask.T, latent_dim)

        #   MI estimator
        self.MI = MIEstimator(latent_dim)
        self.mask = mask.cuda()

        self.latent_mode = 'mu'
        self.latent_dim = latent_dim

        self._initialize_weights()

    def _initialize_weights(self):
        # 遍历模型的每一层并应用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_latent_distribution(self, input_x):

        input_x_rna = input_x[0]
        input_x_atac = input_x[1]
        input_x_TF = input_x[2]

        rna_embedd = self.RNAEncoder(input_x_rna)
        atac_embedd = self.ATACEncoder(input_x_atac)

        #   share feature
        rna_share_mu, rna_share_logvar, rna_share_latent_z = self.ModalShareEncoder(rna_embedd)
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(atac_embedd)

        joint_mu, joint_logvar = product_of_experts([rna_share_mu, atac_share_mu], [rna_share_logvar, atac_share_logvar])
        return {'joint_mu': joint_mu, 'joint_logvar': joint_logvar}

    def forward(self, input_x, batch_id):
        # input_x_rna = torch.concat((input_x[0], batch_id), dim=1)
        # input_x_atac = torch.concat((input_x[1], batch_id), dim=1)
        input_x_rna = input_x[0]
        input_x_atac = input_x[1]
        input_x_TF = input_x[2]

        rna_embedd = self.RNAEncoder(input_x_rna)
        atac_embedd = self.ATACEncoder(input_x_atac)


        #   share feature
        rna_share_mu, rna_share_logvar, rna_share_latent_z = self.ModalShareEncoder(rna_embedd)
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(atac_embedd)

        joint_mu, joint_logvar = product_of_experts([rna_share_mu, atac_share_mu], [rna_share_logvar, atac_share_logvar])
        joint_latent_z = reparameterize(joint_mu, joint_logvar)

        #   private feature
        rna_private_mu, rna_private_logvar, rna_private_latent_z = self.RNAPrivateEncoder(rna_embedd)
        atac_private_mu, atac_private_logvar, atac_private_latent_z = self.ATACPrivateEncoder(atac_embedd)
        if self.latent_mode == 'mu':
            output = {
                'share_embedding': joint_mu,
                'rna_private_embedding': rna_private_mu,
                'atac_private_embedding': atac_private_mu,
                'rna_share_embedding': rna_share_mu,
                'atac_share_embedding': atac_share_mu
            }
        else:
            output = {
                'share_embedding': joint_latent_z,
                'rna_private_embedding': rna_private_latent_z,
                'atac_private_embedding': atac_private_latent_z,
                'rna_share_embedding': rna_share_latent_z,
                'atac_share_embedding': atac_share_latent_z
            }
        return output

    def cross_modal_generation(self, input_x, batch_id):
        # input_x_rna = torch.concat((input_x[0], batch_id), dim=1)
        # input_x_atac = torch.concat((input_x[1], batch_id), dim=1)

        input_x_rna = input_x[0]
        input_x_atac = input_x[1]
        input_x_TF = input_x[2]

        rna_embedd = self.RNAEncoder(input_x_rna)
        atac_embedd = self.ATACEncoder(input_x_atac)

        #   share feature
        rna_share_mu, rna_share_logvar, rna_share_latent_z = self.ModalShareEncoder(rna_embedd)
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(atac_embedd)

        prior_z_rna = torch.randn_like(atac_share_latent_z)
        prior_z_atac = torch.randn_like(rna_share_latent_z)

        recon_batch_id = torch.zeros_like(batch_id).cuda()

        #   cross-modal generation
        rna_recon_cross_from_atac = self.RNADecoder(torch.cat([atac_share_latent_z, prior_z_rna, recon_batch_id], dim=1))
        atac_recon_cross_from_rna = self.ATACDecoder(torch.cat([rna_share_latent_z, prior_z_atac, recon_batch_id], dim=1))

        return rna_recon_cross_from_atac, atac_recon_cross_from_rna

    def rna_generation_from_atac(self, input_x, batch_id):
        input_x_atac = input_x
        atac_embedd = self.ATACEncoder(input_x_atac)

        #   share feature
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(atac_embedd)

        prior_z_rna = torch.randn_like(atac_share_latent_z)
        recon_batch_id = torch.zeros_like(batch_id).cuda()

        #   cross-modal generation
        rna_recon_cross_from_atac = self.RNADecoder(torch.cat([atac_share_latent_z, prior_z_rna, recon_batch_id], dim=1))

        return rna_recon_cross_from_atac

    def atac_generation_from_rna(self, input_x, batch_id):
        input_x_rna = input_x
        rna_embedd = self.ATACEncoder(input_x_rna)

        #   share feature
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(rna_embedd)

        prior_z_atac = torch.randn_like(atac_share_latent_z)
        recon_batch_id = torch.zeros_like(batch_id).cuda()

        #   cross-modal generation
        atac_recon_cross_from_rna = self.ATACDecoder(torch.cat([atac_share_latent_z, prior_z_atac, recon_batch_id], dim=1))

        return atac_recon_cross_from_rna

    def modal_generation_from_latent(self, share_embedding, private_embedding, batch_id, modality_name):
        recon_batch_id = torch.zeros_like(batch_id).cuda()

        latent_z = torch.cat([share_embedding, private_embedding, recon_batch_id], dim=1)
        if modality_name == 'rna':
            modal_generation = self.RNADecoder(latent_z)
        else:
            modal_generation = self.ATACDecoder(latent_z)
        return modal_generation

    def modal_generation_from_latentTF(self, TF_embedding, batch_id, modality_name):
        #
        recon_batch_id = torch.zeros_like(batch_id).cuda()
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(TF_embedding)
        prior_z_atac = torch.randn_like(atac_share_latent_z)
        latent_z = torch.cat([atac_share_latent_z, prior_z_atac, recon_batch_id], dim=1)

        modal_generation = self.RNADecoder(latent_z)

        return modal_generation

    def modal_generation(self, input_x, batch_id):
        input_x_rna = input_x[0]
        input_x_atac = input_x[1]

        rna_embedd = self.RNAEncoder(input_x_rna)
        atac_embedd = self.ATACEncoder(input_x_atac)

        #   share feature
        rna_share_mu, rna_share_logvar, rna_share_latent_z = self.ModalShareEncoder(rna_embedd)
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(atac_embedd)

        joint_mu, joint_logvar = product_of_experts([rna_share_mu, atac_share_mu], [rna_share_logvar, atac_share_logvar])

        #   private feature
        rna_private_mu, rna_private_logvar, rna_private_latent_z = self.RNAPrivateEncoder(rna_embedd)
        atac_private_mu, atac_private_logvar, atac_private_latent_z = self.ATACPrivateEncoder(atac_embedd)

        joint_latent_z = reparameterize(joint_mu, joint_logvar)

        # prior_z_rna = torch.randn_like(atac_share_latent_z)
        # prior_z_atac = torch.randn_like(rna_share_latent_z)

        recon_batch_id = torch.zeros_like(batch_id).cuda()

        #   cross-modal generation
        rna_recon = self.RNADecoder(torch.cat([joint_latent_z, rna_private_latent_z, recon_batch_id], dim=1))
        atac_recon = self.ATACDecoder(torch.cat([joint_latent_z, atac_private_latent_z, recon_batch_id], dim=1))

        return rna_recon, atac_recon

    def compute_loss(self, input_x, batch_id):
        # input_x_rna = torch.concat((input_x[0], batch_id), dim=1)
        # input_x_atac = torch.concat((input_x[1], batch_id), dim=1)

        input_x_rna = input_x[0]
        input_x_atac = input_x[1]
        input_x_TF = input_x[2]

        rna_embedd = self.RNAEncoder(input_x_rna)
        atac_embedd = self.ATACEncoder(input_x_atac)

        # atac_embedd = atac_embedd

        #   share feature
        rna_share_mu, rna_share_logvar, rna_share_latent_z = self.ModalShareEncoder(rna_embedd)
        atac_share_mu, atac_share_logvar, atac_share_latent_z = self.ModalShareEncoder(atac_embedd)


        joint_mu, joint_logvar = product_of_experts([rna_share_mu, atac_share_mu], [rna_share_logvar, atac_share_logvar])
        joint_latent_z = reparameterize(joint_mu, joint_logvar)

        #   private feature
        rna_private_mu, rna_private_logvar, rna_private_latent_z = self.RNAPrivateEncoder(rna_embedd)
        atac_private_mu, atac_private_logvar, atac_private_latent_z = self.ATACPrivateEncoder(atac_embedd)

        #   multimodal latent recon
        rna_latent_z = torch.concat((joint_latent_z, rna_private_latent_z, batch_id), dim=1)
        rna_recon = self.RNADecoder(rna_latent_z)

        atac_latent_z = torch.concat((joint_latent_z, atac_private_latent_z, batch_id), dim=1)
        atac_recon = self.ATACDecoder(atac_latent_z)

        #   recon loss
        rna_recon_loss = Reconstruction_loss(rna_recon, input_x[0], 1, self.RNADataDist)
        atac_recon_loss = Reconstruction_loss(atac_recon, input_x[1], 1, self.ATACDataDist)

        #  self-modal latent recon
        rna_self_latent_z = torch.concat((rna_share_latent_z, rna_private_latent_z, batch_id), dim=1)
        rna_self_recon = self.RNADecoder(rna_self_latent_z)
        atac_self_latent_z = torch.concat((atac_share_latent_z, atac_private_latent_z, batch_id), dim=1)
        atac_self_recon = self.ATACDecoder(atac_self_latent_z)

        #   self recon loss
        rna_self_recon_loss = Reconstruction_loss(rna_self_recon, input_x[0], 1, self.RNADataDist)
        atac_self_recon_loss = Reconstruction_loss(atac_self_recon, input_x[1], 1, self.ATACDataDist)

        #   kl loss
        rna_kl_loss = KL_loss(rna_private_mu, rna_private_logvar, 1)
        atac_kl_loss = KL_loss(atac_private_mu, atac_private_logvar, 1)

        rna_prior_z = torch.randn_like(joint_latent_z)
        atac_prior_z = torch.randn_like(joint_latent_z)

        #   cross-modal generation
        rna_recon_cross_from_atac = self.RNADecoder(torch.cat([atac_share_latent_z, rna_prior_z, batch_id], dim=1))
        atac_recon_cross_from_rna = self.ATACDecoder(torch.cat([rna_share_latent_z, atac_prior_z, batch_id], dim=1))

        rna_cross_recon_loss = Reconstruction_loss(rna_recon_cross_from_atac, input_x[0], 1, self.RNADataDist)
        atac_cross_recon_loss = Reconstruction_loss(atac_recon_cross_from_rna, input_x[1], 1, self.ATACDataDist)

        # Contrastive loss with similarity matrix
        logits = (rna_share_latent_z @ atac_share_latent_z.T) / self.temperature
        rna_similarity = rna_share_latent_z @ rna_share_latent_z.T
        atac_similarity = atac_share_latent_z @ atac_share_latent_z.T
        targets = F.softmax(
            (rna_similarity + atac_similarity) / 2 * self.temperature, dim=-1
        )
        rna_loss = F.cross_entropy(logits, targets, reduction='mean')
        atac_loss = F.cross_entropy(logits.T, targets.T, reduction='mean')
        multimodal_contrastive_loss = (rna_loss + atac_loss)

        #   contrastive loss
        rna_contrastive_loss = self.contrastive_loss(rna_share_latent_z, atac_share_latent_z, rna_private_latent_z)
        atac_contrastive_loss = self.contrastive_loss(atac_share_latent_z, rna_share_latent_z, atac_private_latent_z)
        contrastive_loss = rna_contrastive_loss + atac_contrastive_loss

        share_kl_loss = KL_loss(joint_mu, joint_logvar, 1) + (KL_loss(rna_share_mu, rna_share_logvar, 1) +
                                                              KL_loss(atac_share_mu, atac_share_logvar, 1))

        PIB_mimin = self.MI.learning_loss(rna_private_latent_z, atac_private_latent_z, joint_latent_z, rna_share_latent_z, atac_share_latent_z)
        PIB_mi = self.MI(rna_private_latent_z, atac_private_latent_z, joint_latent_z, rna_share_latent_z, atac_share_latent_z)

        loss = (
                (rna_self_recon_loss + atac_self_recon_loss) +
                (rna_recon_loss + atac_recon_loss) +
                (rna_cross_recon_loss + atac_cross_recon_loss) +
                self.kl_weight * (share_kl_loss + rna_kl_loss + atac_kl_loss) +
                self.con_weight * (contrastive_loss + multimodal_contrastive_loss) +
                self.con_weight * (PIB_mi + PIB_mimin)
                #0.01 * torch.sum(torch.abs(self.ATACDecoder.weight[:, :self.latent_dim]))
                #0.01 * torch.sum(torch.abs(joint_latent_z)) / self.latent_dim

                # reg_loss(joint_latent_z, input_x_TF, joint_latent_z.shape[0])
        )
        return loss

    @staticmethod
    def contrastive_loss(anchor, positive, negative, margin=0.5):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))
        return loss

    @staticmethod
    def contrastive_loss_with_similarity(anchor, positive, negative, margin=1.0):
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        loss = torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))
        loss = - torch.log(loss).mean()
        # loss = torch.mean(torch.relu(neg_sim - pos_sim + margin))
        return loss


class explainModel(nn.Module):
    def __init__(self,
                 scModel: scMulti,
                 mode,
                 latent_dim,
                 gene_ids,
                 cluster_num=None,
                 classifier=None,
                 cluster_predictor=None
                 ):
        super(explainModel, self).__init__()
        self.scDM = scModel
        self.mode = mode
        self.gene_ids = gene_ids
        if cluster_num is not None:
            self.classifier = LinearLayer(latent_dim, cluster_num)

    def forward(self, x):
        rna_input, atac_input, batch_id = torch.split(x, [self.scDM.input_dims[0], self.scDM.input_dims[1], self.scDM.batch_dims], dim=1)

        #   cross modal translation
        if self.mode == 'RNA2ATAC':
            atac_cross_recon = self.scDM.atac_generation_from_rna(rna_input, batch_id)
            return atac_cross_recon

        if self.mode == 'ATAC2RNA':
            rna_cross_recon = self.scDM.rna_generation_from_atac(atac_input, batch_id)
            return rna_cross_recon[:, self.gene_ids]

        #   regulatory generation
        if self.mode == 'RNARegulatory':
            rna_recon, _ = self.scDM.modal_generation([rna_input, atac_input], batch_id)
            return rna_recon[:, self.gene_ids]

        #   cluster mark gene
        if self.mode == 'joint_share':
            output = self.scDM([rna_input, atac_input], batch_id)['share_embedding']
            # output = output.detach().cpu().numpy()
            # prediction_output = self.classifier(output)
            return output[:, 4]

        if self.mode == 'rna_private':
            output = self.scDM([rna_input, atac_input], batch_id)['rna_private_embedding']
            # output = output.detach().cpu().numpy()
            # prediction_output = self.classifier(output)
            return output

        if self.mode == 'atac_private':
            output = self.scDM([rna_input, atac_input], batch_id)['atac_private_embedding']
            output = output.detach().cpu().numpy()
            prediction_output = self.classifier(output)
            return prediction_output


class explainModelLatentZ(nn.Module):
    def __init__(self,
                 scModel: scMulti,
                 mode,
                 latent_dim,
                 dimension_num=None
                 ):
        super(explainModelLatentZ, self).__init__()
        self.scDM = scModel
        self.mode = mode
        self.latent_dim = latent_dim
        self.dimension_num = dimension_num

    def forward(self, x):

        share_embedding, private_embedding, batch_id = torch.split(x,[self.latent_dim, self.latent_dim, self.scDM.batch_dims], dim=1)
        output = self.scDM.modal_generation_from_latent(share_embedding, private_embedding, batch_id, self.mode)
        # loss = F.mse_loss(output, test_rna_, reduction='none').mean(1).view(-1, 1)
        if self.dimension_num is not None:
            output = output[:, [self.dimension_num]]
        return output

        #   cross modal translation


class explainModelLatentTF(nn.Module):
    def __init__(self,
                 scModel: scMulti,
                 mode,
                 latent_dim,
                 dimension_num=None
                 ):
        super(explainModelLatentTF, self).__init__()
        self.scDM = scModel
        self.mode = mode
        self.latent_dim = latent_dim
        self.dimension_num = dimension_num

    def forward(self, x):

        TF_embedding, batch_id = torch.split(x,[514, self.scDM.batch_dims], dim=1)
        output = self.scDM.modal_generation_from_latentTF(TF_embedding, batch_id, self.mode)

        if self.dimension_num is not None:
            output = output[:, [self.dimension_num]]
        return output


def calculate_r_squared_torch(y_true, y_pred):
    # 计算总平方和 (TSS)
    tss = torch.sum((y_true - torch.mean(y_true, axis=0))**2, axis=0)
    # 计算残差平方和 (RSS)
    rss = torch.sum((y_true - y_pred)**2, axis=0)
    # 计算R平方值
    r_squared = 1 - (rss / tss)
    return r_squared

