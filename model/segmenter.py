import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from model.clip import build_model
from utils.dataset import tokenize
from .layers import FPN, V2LFusion, L2VFusion, Projector, VisProjector, TransformerDecoder, VisionLanguageCrossFusion
from .loss import mask_to_counter, dist_loss

def att_kd_mse(att_s, att_t):
    att_kd_loss = F.mse_loss(att_s, att_t.detach())
    return att_kd_loss

def att_kd_kl(att_s, att_t):
    att_s = att_s.permute(0, 2, 1)
    att_t = att_t.permute(0, 2, 1)
    att_kd_loss = F.kl_div(F.log_softmax(att_s, -1), F.softmax(att_t, -1).detach(), reduction='sum')
    return att_kd_loss

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)

        # L2V Fusion
        # self.l2v = L2VFusion(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)

        # V2L Fusion
        # self.v2l = V2LFusion(uni_channels=cfg.vis_dim)

        # VisionLanguageCrossFusion
        self.vlcf = VisionLanguageCrossFusion(num_layers=cfg.vlcf_num_layers, feat_channels=cfg.vis_dim)
        
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        # self.vis_proj = VisProjector(676, cfg.word_dim, cfg.vis_dim // 2, 3)
        # self.norm = nn.LayerNorm(cfg.word_dim)

    def forward(self, img, word, mask=None):
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        vis, v_g = self.backbone.encode_image(img)      # (C3, C4, C5),   (b, 1024, 1, 1)
        word, t_g = self.backbone.encode_text(word)     # (b, L, 1024),   (b, 1024)

        # v2l
        # word = self.v2l(word, v_g)
        # l2v
        # fq = self.l2v(vis, t_g)
        fq = self.neck(vis, t_g)        # b, 512, 26, 26 (C4)

        b, c, h, w = fq.size()
        # VisionLanguageCrossFusion
        word, fq = self.vlcf(word, fq)
        # fv = fq.reshape(b, c, h, w)
        
        # ---------- Feat Distill ----------
        # prefq = fq
        # fq = self.decoder(fq, word, pad_mask)   # w Decoder
        # -------- Attention Distill -------
        fq, att_maps = self.decoder(fq, word, pad_mask)    # att_maps   return_intermediate = False
        # --------- Query Distill ----------
        # mid_feats = self.decoder(fq, word, pad_mask)    # intermediate: [(32, 512, 676); (32, 512, 676); (32, 512, 676);]
        # fq = mid_feats[-1]

        fq = fq.reshape(b, c, h, w)
        pred = self.proj(fq, t_g)   # b, 1, 104, 104

        # cap = self.vis_proj(fq, pred, fv)
        # pred2 = self.proj(fq, cap)   # b, 1, 104, 104

        # ---------------- Feat Visualization ----------------
        # prefq_vis = torch.sum(torch.abs(F.interpolate(prefq, (416, 416))), dim=1).squeeze(0)
        # prefq_vis = prefq_vis.cpu().detach().numpy()
        # fq_vis = torch.sum(torch.abs(F.interpolate(fq, (416, 416))), dim=1).squeeze(0)
        # fq_vis = fq_vis.cpu().detach().numpy()
        # plt.imsave('./Visualization/FeatShow/visual_feat.png', prefq_vis, cmap='RdBu_r')
        # plt.imsave('./Visualization/FeatShow/cross_feat.png', fq_vis, cmap='RdBu_r')

        # ---------------- Attention Visualization ----------------
        # for i, att_map in enumerate(att_maps):
        #     att_map = att_map.squeeze(0)[:,1]
        #     att_map = att_map.reshape(26, 26)
        #     att_map = att_map.cpu().detach().numpy()
        #     plt.imsave('./Visualization/AttShow/att_map{}.png'.format(i), att_map, cmap='RdBu_r')

        # ---------------- Query Visualization ----------------
        # for i, mid_feat in enumerate(mid_feats):
        #     mid_feat = mid_feat.reshape(-1, 512, 26, 26)
        #     mid_feat = torch.sum(torch.abs(F.interpolate(mid_feat, (416, 416))), dim=1).squeeze(0)
        #     mid_feat = mid_feat.cpu().detach().numpy()
        #     plt.imsave('./Visualization/QueryShow/query_feat{}.png'.format(i), mid_feat, cmap='RdBu_r')

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            mask_loss = F.binary_cross_entropy_with_logits(pred, mask)
            
            # ---------- Semantic Consistency Constraints-----------
            # t_g = self.norm(t_g)
            # # sem_loss = F.mse_loss(cap, t_g.detach()) / 1024                                   # - 1 -
            # cos_sim = torch.cosine_similarity(cap, t_g.detach(), dim=-1)                        # - 2 -
            # cos_sim_loss = 1 - cos_sim.mean(dim=0)
      
            # cons_loss = F.binary_cross_entropy_with_logits(pred2, pred.detach())              # - 3 -

            # ---------- Feat Distill ----------
            # feat_kd_loss = F.mse_loss(prefq, fq.detach())
            # -------- Attention Distill ------- 0, 1, -1
            ## 4 layer
            # att_kd_loss = att_kd_mse(att_maps[0], att_maps[1]) + att_kd_mse(att_maps[0], att_maps[2]) + att_kd_mse(att_maps[0], att_maps[-1]) + att_kd_mse(att_maps[1], att_maps[2]) + att_kd_mse(att_maps[1], att_maps[-1]) + att_kd_mse(att_maps[2], att_maps[-1])
            # loc_loss = (att_kd_loss / 6) * 0.1
            ## 3 layer
            att_kd_loss = att_kd_mse(att_maps[0], att_maps[1]) + att_kd_mse(att_maps[0], att_maps[-1]) + att_kd_mse(att_maps[1], att_maps[-1])
            loc_loss = (att_kd_loss / 3) * 3
            # --------- Query Distill ----------
            # query_kd_loss = att_kd_mse(mid_feats[0], mid_feats[1]) + att_kd_mse(mid_feats[0], mid_feats[-1]) + att_kd_mse(mid_feats[1], mid_feats[-1])
            # loss = loss + (query_kd_loss / 3) * 0.1

            loss = mask_loss + loc_loss

            return pred.detach(), mask, loss
        else:
            return pred.detach()
