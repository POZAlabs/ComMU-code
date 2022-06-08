import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs=None, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        if cutoffs is None:
            cutoffs = []

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(
                torch.zeros(self.n_clusters, self.d_embed)
            )
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()


        for i in range(len(self.cutoffs)):
            if d_proj != d_embed:
                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
            else:
                self.out_projs.append(None)

        self.out_layers.append(nn.Linear(d_embed, n_token))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)

        return logit

    def forward(self, hidden, target, keep_order=False):
        """
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        """

        if hidden.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )

        if self.n_clusters == 0:
            logit = self._compute_logit(
                hidden,
                self.out_layers[0].weight,
                self.out_layers[0].bias,
                self.out_projs[0],
            )
            nll = (
                -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            )
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                weight_i = self.out_layers[0].weight[l_idx:r_idx]
                bias_i = self.out_layers[0].bias[l_idx:r_idx]

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(
                        hidden_i, weight_i, bias_i, proj_i
                    )
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(
                        1, target_i[:, None]
                    ).squeeze(1)

                if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset: offset + logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)  # Outer product
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, inp):
        ##### positionwise feed-forward
        core_out = self.CoreNet(inp)

        ##### residual connection + layer normalization
        output = self.layer_norm(inp + core_out)

        return output


# Main attention class with all lengths
class RelMultiHeadAttn(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=0,
            tgt_len=None,
            mem_len=None,
            use_qkv=True,
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        if use_qkv:
            self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        else:
            self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
            self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)  # Split into k and v later

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros(
            (x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


# Default attention layer used
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(
            *args, **kwargs
        )  # ext_len passed here

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(
            qlen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->bnij", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum(
            "ibnd,jnd->bnij", (rr_head_q, r_head_k)
        )  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float("inf"))

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("bnij,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        output = self.layer_norm(w + attn_out)

        return output


# Default attention layer used
class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )

        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(
            dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems
        )

        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj):
        """

        :param n_token: number of tokens in vocab
        :param d_embed: dimension of embedding
        :param d_proj: dimension of embedding projection (unused here since d_proj = d_model)
        """

        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = [n_token]
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        self.emb_layers.append(
            nn.Embedding(n_token, d_embed, sparse=False)
        )

        if d_proj != d_embed:
            self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))

    def forward(self, inp):
        if len(inp.shape) == 2:
            embed = self.emb_layers[0](inp)
        else:
            embed = torch.matmul(inp, self.emb_layers[0].weight)

        if self.d_proj != self.d_embed:
            embed = F.linear(embed, self.emb_projs[0])

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(
            self,
            cfg,
            vocab
    ):
        n_layer = cfg.MODEL.num_layers
        n_head = cfg.MODEL.num_heads
        d_model = cfg.MODEL.units
        d_head = cfg.MODEL.units // cfg.MODEL.num_heads
        d_inner = cfg.MODEL.inner_size
        dropout = cfg.MODEL.dropout
        dropatt = cfg.MODEL.attention_dropout
        d_embed = cfg.MODEL.units
        tgt_len = cfg.TRAIN.tgt_length
        mem_len = cfg.TRAIN.mem_length
        same_length = cfg.MODEL.same_length
        clamp_len = cfg.MODEL.clamp_len

        super(MemTransformerLM, self).__init__()
        self.cfg = cfg
        self.n_token = len(vocab)
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(
            self.n_token, d_embed, d_model)

        self.drop = nn.Dropout(dropout)
        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.max_klen = tgt_len + mem_len

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    dropout,
                    tgt_len=tgt_len,
                    mem_len=mem_len,
                    dropatt=dropatt,
                )
            )
        self.crit = ProjectedAdaptiveLogSoftmax(
            self.n_token, d_embed, d_model
        )

        for i in range(len(self.crit.out_layers)):
            self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

        self.same_length = same_length
        self.clamp_len = clamp_len

        self.detach_mems_grad = True
        self._create_params()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len

    def init_mems(self, n_layers):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems = torch.empty(n_layers + 1, 0, dtype=param.dtype,
                               device=param.device)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen, reset_mems=None):

        # The idea is that randomization from shuffling will have be equivalent to memory resetting

        if mems is None:
            return None

        assert len(hids) == len(mems)
        # mems is not the same as self.mem_len

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            stacked = torch.stack(hids)

            if mems.numel():
                cat = torch.cat([mems, stacked], dim=1)
            else:
                cat = stacked

            if self.detach_mems_grad:
                new_mems = cat[:, beg_idx:end_idx].detach()
            else:
                new_mems = cat[:, beg_idx:end_idx].detach()

        return new_mems

    def _forward(self, dec_inp, reset_mems, mems=None):

        qlen, bsz = dec_inp.size()[0], dec_inp.size()[1]
        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        # Generate the mask between query and all the keys
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen

        if reset_mems is None:
            indices = torch.BoolTensor(dec_inp.shape[1]).fill_(False)
        else:
            indices = reset_mems

        if self.same_length:
            dec_attn_mask = ((
                                     torch.triu(all_ones, 1 + mlen)
                                     + torch.tril(all_ones, -mask_shift_len)
                             ).bool()[
                             :, :
                             ]).repeat(len(indices), 1, 1)  # -1
        else:
            dec_attn_mask = (torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen
            ).bool()[:, :]).repeat(len(indices), 1, 1)

        dec_attn_mask[indices, :, :mlen] = 1


        hids = []
        pos_seq = torch.arange(
            klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype
        )
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(
                core_out,
                pos_emb,
                self.r_w_bias,
                self.r_r_bias,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
            )
            hids.append(core_out)
        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen, reset_mems)
        return core_out, new_mems

    def forward_generate(self, data, mems):

        if mems is None:
            mems = self.init_mems(self.n_layer)

        tgt_len = data.size(0)
        batch_size = data.size(1)

        hidden, new_mems = self._forward(data, None, mems=mems)

        pred_hid = hidden[-tgt_len:]

        assert self.crit.n_clusters == 0

        logits = self.crit._compute_logit(
            pred_hid.view(-1, pred_hid.size(-1)),
            self.crit.out_layers[0].weight,
            self.crit.out_layers[0].bias,
            self.crit.out_projs[0],
        )
        logits = logits.view(tgt_len, batch_size, -1)

        return (logits, new_mems)

    def forward_generate_gumbel(self, data, temperature, mems):

        from torch.autograd import Variable

        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape).cuda()
            return -Variable(torch.log(-torch.log(U + eps) + eps))

        def gumbel_softmax_sample(logits, temperature):
            y = logits + sample_gumbel(logits.size())
            return F.softmax(y / temperature, dim=-1)

        def gumbel_softmax(logits, temperature):
            """
            input: [*, n_class]
            return: [*, n_class] an one-hot vector
            """
            y = gumbel_softmax_sample(logits, temperature)
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            return (y_hard - y).detach() + y

        if mems is None:
            mems = self.init_mems(self.n_layer)

        tgt_len = data.size(0)
        batch_size = data.size(1)
        hidden, new_mems = self._forward(data, None, mems=mems)

        pred_hid = hidden[-tgt_len:]

        assert self.crit.n_clusters == 0

        logits = self.crit._compute_logit(
            pred_hid.view(-1, pred_hid.size(-1)),
            self.crit.out_layers[0].weight,
            self.crit.out_layers[0].bias,
            self.crit.out_projs[0],
        )
        logits = gumbel_softmax(
            logits.view(tgt_len, batch_size, -1), temperature=temperature
        )

        return (logits, new_mems)

    def forward(self, data, target, reset_mems, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if mems is None:
            mems = self.init_mems(self.n_layer)

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, reset_mems, mems=mems)

        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

        return (loss, new_mems)
