import torch
import torch.nn as nn
import torch.nn.functional as F

import math




EPS = torch.tensor(1e-15)


class LocalHyperGATlayer(nn.Module):
    def __init__(self, dim, layer, alpha, dropout=0., bias=False, act=True):
        super(LocalHyperGATlayer, self).__init__()
        self.dim = dim
        self.layer = layer
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.act = act

        if self.act:
            self.acf = torch.relu

        # Parameters
        # node->edge->node
        self.a10 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.a20 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))

        self.reset_parameters()

        self.leakyrelu = nn.LeakyReLU(self.alpha)


        self.trans_layer = nn.Linear(self.dim + self.dim, self.dim)

        self.dropout = nn.Dropout(self.dropout, True)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        self.a10.data.uniform_(-stdv, stdv)
        self.a20.data.uniform_(-stdv, stdv)



    def generate_hye_emb(self, node_embedding, hye_len, hye_node):
        zeros = torch.zeros(1, list(node_embedding.shape)[1])
        node_embedding = torch.cat([zeros, node_embedding], 0)
        seq_h = node_embedding[hye_node]

        hs = torch.div(torch.sum(seq_h, 1), hye_len + EPS)
        return hs

    def forward(self, hyg_ls, node_embedding):
        """
        Input: hidden:(Batchsize, N, latent_dim), incidence matrix H:(batchsize, N, num_edge), session cluster s_c:(Batchsize, 1, latent_dim)
        Output: updated hidden:(Batchsize, N, latent_dim)
        """
        hyg, hye_len, hye_node = hyg_ls[0], hyg_ls[1], hyg_ls[2]


        N = hyg.shape[0]  # node num
        edge_num = hyg.shape[1]  # edge num


        e_emb = self.generate_hye_emb(node_embedding, hye_len, hye_node)

        h_emb = node_embedding
        h_embs = [node_embedding]

        hye_emb = e_emb
        hye_embs = [e_emb]


        for i in range(self.layer):
            # node2edge
            edge_c_in = hye_emb.unsqueeze(0).expand(N, -1, -1)  # (N, edge_num, latent_dim)
            h_4att0 = h_emb.unsqueeze(1).expand(-1, edge_num, -1)  # (N, edge_num, latent_dim)

            feat = edge_c_in * h_4att0

            atts10 = self.leakyrelu(torch.matmul(feat, self.a10).squeeze(-1))  # (N, edge_num)

            zero_vec = -9e15 * torch.ones_like(hyg)
            alpha1 = torch.where(hyg.eq(1), atts10, zero_vec)

            alpha1 = F.softmax(alpha1, dim=0)  # (N, edge_num)

            hye_feat = torch.matmul(alpha1.transpose(0, 1), h_emb)  # (edge_num, latent_dim)

            edge = torch.cat((hye_feat, hye_emb), dim=1)

            edge = self.trans_layer(edge)


            # edge2node
            edge_in = edge.unsqueeze(0).expand(N, -1, -1)  # (N, edge_num, latent_dim)
            h_4att1 = h_emb.unsqueeze(1).expand(-1, edge_num, -1)  # (N, edge_num, latent_dim)

            feat_e2n = edge_in * h_4att1

            atts20 = self.leakyrelu(torch.matmul(feat_e2n, self.a20).squeeze(-1))  # (N, edge_num)
            alpha2 = torch.where(hyg.eq(1), atts20, zero_vec)


            alpha2 = F.softmax(alpha2, dim=1)  # (N, edge_num)
            h_emb = torch.matmul(alpha2, edge)  # (N, latent_dim)

            h_emb = self.dropout(h_emb)

            h_embs.append(h_emb)


            hye_emb = self.generate_hye_emb(h_emb, hye_len, hye_node)
            hye_embs.append(hye_emb)


        h_embs = torch.stack(h_embs, dim=1)
        h_out = torch.mean(h_embs, dim=1)

        hye_embs = torch.stack(hye_embs, dim=1)
        hye_out = torch.mean(hye_embs, dim=1)

        return h_out, hye_out





class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc23 = nn.Linear(in_channels // 4, in_channels // 8)
        self.batch23 = nn.BatchNorm1d(in_channels // 8)
        self.fc3 = nn.Linear(in_channels // 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, circ_embed, mi_embed, circRNA_id, miRNA_id):
        h_0 = torch.cat((circ_embed[circRNA_id, :], mi_embed[miRNA_id, :]), 1)
        h_1 = torch.tanh(self.fc1(h_0))
        h_1 = self.batch1(h_1)
        h_2 = torch.tanh(self.fc2(h_1))
        h_2 = self.batch2(h_2)
        h_23 = torch.tanh(self.fc23(h_2))
        h_23 = self.batch23(h_23)
        h_3 = self.fc3(h_23)
        return torch.sigmoid(h_3.squeeze(dim=1))







class BioEncoder(nn.Module):
    def __init__(self, dim_circ, dim_mi, output):
        super(BioEncoder, self).__init__()
        self.circ_layer1 = nn.Linear(dim_circ, output)
        self.batch_circ1 = nn.BatchNorm1d(output)

        self.mi_layer1 = nn.Linear(dim_mi, output)
        self.batch_mi1 = nn.BatchNorm1d(output)
        self.relu = nn.ReLU()
        self.reset_para()


    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, circRNA_feature, miRNA_feature):
        x_circ = self.circ_layer1(circRNA_feature)
        x_circ = self.batch_circ1(F.relu(x_circ))

        x_mi = self.mi_layer1(miRNA_feature)
        x_mi = self.batch_mi1(F.relu(x_mi))

        return x_circ, x_mi



class gate_layer(nn.Module):
    def __init__(self, emb_size):
        super(gate_layer, self).__init__()
        self.emb_size = emb_size

        self.w2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u2 = nn.Linear(self.emb_size, self.emb_size, bias=True)


        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, f_interaction1, f_interaction2, graph):

        r2 = torch.sigmoid(self.w2(f_interaction1) + self.u2(f_interaction2))

        m1 = f_interaction1 * r2 + (1 - r2) * f_interaction2

        m2 = torch.mm(graph, m1)


        return torch.cat([m1, m2], dim=1)

class HGNN(nn.Module):
    def __init__(self, n_node_c, n_node_m, emb_size, out_size, layers_h,
                 hy_cm, hy_mc, cc_hyper_graph, mm_hyper_graph, cc_D, mm_D):
        super(HGNN, self).__init__()
        self.n_node_c = n_node_c
        self.n_node_m = n_node_m

        self.bio_encoder = BioEncoder(n_node_c, n_node_m, emb_size)

        self.emb_size = emb_size
        self.out_size = out_size
        self.layers_h = layers_h

        self.cc_hyper_graph = cc_hyper_graph
        self.mm_hyper_graph = mm_hyper_graph
        self.cc_D = cc_D
        self.mm_D = mm_D


        self.hy_cm = hy_cm
        self.hy_mc = hy_mc

        self.hyg_aggs1 = LocalHyperGATlayer(self.emb_size, self.layers_h, 0.2, 0.5)

        self.interact_c = gate_layer(self.out_size)
        self.interact_m = gate_layer(self.out_size)


        self.decoder_cmi = Decoder(out_size * 8)


    def forward(self,  circ_feature, mi_feature, circ_id, mi_id):

        circRNA_embedding, miRNA_embedding = self.bio_encoder(circ_feature, mi_feature)

        cc_DA = torch.mm(self.cc_D, self.cc_hyper_graph)
        mm_DA = torch.mm(self.mm_D, self.mm_hyper_graph)


        circRNA_emb1, hye_miRNA_emb1 = self.hyg_aggs1(self.hy_mc, circRNA_embedding)
        miRNA_emb1, hye_circRNA_emb2 = self.hyg_aggs1(self.hy_cm, miRNA_embedding)

        hs_c1 = torch.mm(cc_DA, hye_circRNA_emb2)
        hs_m1 = torch.mm(mm_DA, hye_miRNA_emb1)

        hs_c2 = torch.mm(cc_DA, circRNA_emb1)
        hs_m2 = torch.mm(mm_DA, miRNA_emb1)



        hc1 = self.interact_c(circRNA_emb1, hs_c2, cc_DA)
        hc2 = self.interact_c(hye_circRNA_emb2, hs_c1, cc_DA)

        hm1 = self.interact_m(miRNA_emb1, hs_m2, mm_DA)
        hm2 = self.interact_m(hye_miRNA_emb1, hs_m1, mm_DA)

        hc = torch.cat([hc1, hc2], dim=1)
        hm = torch.cat([hm1, hm2], dim=1)

        res_cmi = self.decoder_cmi(hc, hm, circ_id, mi_id)


        return res_cmi






