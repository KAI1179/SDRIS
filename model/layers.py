import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x

class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        # x: b, 512, 26, 26
        # word: b, 512
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> {(b, 256, 3, 3); (b)}
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d : 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out

class VisProjector(nn.Module):
    def __init__(self, cap_dim=676, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis_up = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # vis feat fusion
        self.vis_fuse = nn.Sequential(
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # embed projector
        self.emb_up = nn.Sequential(  # channel
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # weight projector
        out_dim = 1 * in_dim * kernel_size + 1
        self.updim = nn.Linear(in_dim, out_dim)
        self.downdim = nn.Linear(cap_dim, word_dim)
        self.mask_norm1 = nn.LayerNorm(in_dim)
        self.mask_norm = nn.LayerNorm(out_dim)
        self.norm = nn.LayerNorm(word_dim)

    def forward(self, embed, mask, f_v):
        # mask: b, 1, 104, 104      v_g: b, 
        # word: b, 512
        f_v = self.vis_up(f_v)
        B, C, H, W = f_v.size()
        # ------------ mask + img_feat -------------
        # f_v = f_v.view(B, C, H * W)
        # mask = mask.view(B, 1, -1)
        # mask = torch.matmul(mask, f_v.permute(0, 2, 1))
        mask = torch.cat([mask * f_v, f_v], 1)
        mask = self.vis_fuse(mask)      # B, C, H, W
        mask = mask.view(B, C, -1)      # B, C, H * W
        mask = torch.sum(mask, dim=-1)
        mask = self.mask_norm1(mask)
        mask = self.updim(mask)
        mask = self.mask_norm(mask)
        embed = self.emb_up(embed)
        embed = embed.reshape(1, B * C, -1)
        weight, bias = mask[:, :-1], mask[:, -1]
        weight = weight.reshape(B, C, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        cap = F.conv1d(embed,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        cap = cap.squeeze(0)
        cap = self.downdim(cap)
        cap = self.norm(cap)

        return cap

class VisionLanguageCrossFusion(nn.Module):
    def __init__(self, 
                num_layers,
                feat_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            VisionLanguageCrossFusionLayer(feat_channels=feat_channels) for _ in range(num_layers)
        ])

    def forward(self, text, vis):
        for layer in self.layers:
            text, vis = layer(text, vis)

        return text, vis


class VisionLanguageCrossFusionLayer(nn.Module):
    def __init__(self, 
                feat_channels=512):
        super().__init__()
        self.channels = feat_channels

        # Middle projection 
        self.mid_proj_l = nn.Linear(self.channels, self.channels)
        # self.mid_norm_l = nn.Sequential(nn.BatchNorm1d(self.channels), nn.ReLU(True))
        self.mid_proj_v = nn.Linear(self.channels, self.channels)
        # self.mid_norm_v = nn.Sequential(nn.BatchNorm1d(self.channels), nn.ReLU(True))

        # Out projection        
        self.out_proj_l = nn.Linear(self.channels, self.channels)
        # self.out_norm_l = nn.Sequential(nn.BatchNorm1d(self.channels), nn.ReLU(True))
        self.out_proj_v = nn.Linear(self.channels, self.channels)
        # self.out_norm_v = nn.Sequential(nn.BatchNorm1d(self.channels), nn.ReLU(True))

        self.coordconv = nn.Sequential(
            CoordConv(512, 512, 3, 1),
            conv_layer(512, 512, 3, 1))

    def forward(self, l, v):
        # v shape: (B, 512, H, W)
        # l input shape: (B, L, 1024)
        B, C, H, W = v.size()
        v = v.reshape(B, C, H * W)
        # l = l.permute(0, 2, 1)

        vl_mat = torch.matmul(v.permute(0, 2, 1), l.permute(0, 2, 1))  # (B, H*W, N)
        lv_mat = torch.matmul(l, v)  # (B, N, H*W)
        vl_wight = F.softmax(vl_mat, dim=-1)  # (B, HW, N)
        lv_wight = F.softmax(lv_mat, dim=-1)  # (B, N, HW)
        vl = torch.matmul(vl_wight, l)  # (B, H*W, C)
        lv = torch.matmul(lv_wight, v.permute(0, 2, 1))  # (B, N, C)
        vl = self.mid_proj_v(vl)
        # vl = self.mid_norm_v(vl.permute(0, 2, 1)).permute(0, 2, 1)
        lv = self.mid_proj_l(lv)
        # lv = self.mid_norm_l(lv.permute(0, 2, 1)).permute(0, 2, 1)

        lv_mat2 = torch.matmul(lv, vl.permute(0, 2, 1))  # (B, N, HW)
        vl_mat2 = torch.matmul(vl, lv.permute(0, 2, 1))  # (B, HW, N)
        lv_wight2 = F.softmax(lv_mat2, dim=-1)  # (B, N, HW)
        vl_wight2 = F.softmax(vl_mat2, dim=-1)  # (B, HW, N)
        lv2 = torch.matmul(lv_wight2, vl)  # (B, N, C)
        vl2 = torch.matmul(vl_wight2, lv)  # (B, HW, C)

        lv2 = self.out_proj_l(lv2 + l)
        # lv2 = self.out_norm_l(lv2.permute(0, 2, 1)).permute(0, 2, 1)
        vl2 = self.out_proj_v(vl2 + v.permute(0, 2, 1)).permute(0, 2, 1)
        # vl2 = self.out_norm_v(vl2.permute(0, 2, 1)).reshape(B, C, H, W)
        vl2 = vl2.reshape(B, C, H, W)

        vl2 = self.coordconv(vl2)

        return lv2, vl2


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        # vis: b, 512, h, w
        # txt: b, L, 512
        # pad_mask: b, L
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        att_maps = []
        for layer in self.layers:
            output, att_map = layer(output, txt, vis_pos, txt_pos, pad_mask)
            att_maps.append(att_map)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output, att_maps
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention wo Weight Output
        vis2 = self.norm2(vis)
        vis2, att_map = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                            key=self.with_pos_embed(txt, txt_pos),
                                            value=txt,
                                            key_padding_mask=pad_mask)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)

        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis, att_map


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj_t = linear_layer(in_channels[2], out_channels[2])
        self.txt_proj_m = linear_layer(in_channels[2], out_channels[2])
        self.txt_proj_b = linear_layer(in_channels[2], out_channels[1])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]), nn.ReLU(True))
        self.norm_layer_c4 = nn.Sequential(nn.BatchNorm2d(out_channels[2]), nn.ReLU(True))
        self.norm_layer_c3 = nn.Sequential(nn.BatchNorm2d(out_channels[1]), nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        # self.coordconv = nn.Sequential(
        #     CoordConv(out_channels[1], out_channels[1], 3, 1),
        #     conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state_t = self.txt_proj_t(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state_t)
        # fusion 2: b, 512, 26, 26
        state_m = self.txt_proj_m(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        v4 = self.norm_layer_c4(v4 * state_m)  # zhy
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        state_b = self.txt_proj_b(state).unsqueeze(-1).unsqueeze(-1)  # b, 512, 1, 1
        v3 = self.norm_layer_c3(v3 * state_b)  # zhy
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        # fq = self.coordconv(fq)
        return fq  # b, 512, 26, 26


class V2LFusion(nn.Module):
    def __init__(self, uni_channels):
        super().__init__()
        self.v_proj = conv_layer(uni_channels * 2, uni_channels, 1, 0)
        self.l_proj = nn.Linear(uni_channels, uni_channels)
        # self.norm_layer = nn.Sequential(nn.BatchNorm1d(uni_channels), nn.ReLU(True))

    def forward(self, word, v_g):
        # word: (B, L, 512)
        # v_g:  (B, 1024, 1, 1)
        # word = word.permute(0, 2, 1)        # (b, 512, L)
        v_g = self.v_proj(v_g).squeeze(-1)  # (b, 512, 1)

        lv_vector = torch.matmul(word, v_g)  # (B, L, 1)
        lv_wight = F.softmax(lv_vector, dim=1)  # (B, L, 1)
        lv = torch.matmul(lv_wight, v_g.permute(0, 2, 1))  # (B, L, C)
        lv = self.l_proj(lv + word)

        return lv


class L2VFusion(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(L2VFusion, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[1])
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[1]), nn.ReLU(True))

        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)

        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1], out_channels[1], 1, 0)

        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1], out_channels[1], 1, 0)

        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)

    def forward(self, imgs, t_g):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        t_g = self.txt_proj(t_g).unsqueeze(-1)  # b, 512, 1
        v3, v4, v5 = imgs

        f5 = self.f1_v_proj(v5)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')

        f4 = self.f2_v_proj(v4)
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))

        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))

        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)

        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)

        B, C, H, W = fq.size()
        fq = fq.reshape(B, C, H * W)
        vl_vector = torch.matmul(fq.permute(0, 2, 1), t_g)  # (B, HW, 1)
        vl_wight = F.softmax(vl_vector, dim=1)  # (B, HW, 1)
        vl = torch.matmul(vl_wight, t_g.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, HW)
        vl = self.norm_layer(vl.reshape(B, C, H, W))

        return vl  # b, 512, 26, 26
