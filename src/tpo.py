import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers

from src import contriever, dist_utils, utils


class TimeDPOKLQ(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(TimeDPOKLQ, self).__init__()

        self.queue_size = opt.queue_size
        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        # apply the encoder on keys in train mode
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k

        retriever, tokenizer = self._load_retriever(
            opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
        )

        self.tokenizer = tokenizer
        self.encoder_q = retriever
        self.encoder_k = copy.deepcopy(retriever)
        self.encoder_ref = copy.deepcopy(retriever)
        self.beta = 0.3

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(
            opt.projection_size, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _momentum_update_k_encoder(self):
        """
        Update of the positive encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                param_q.data * (1.0 - self.momentum)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # for simplicity
        assert self.queue_size % batch_size == 0, f"{batch_size}, {self.queue_size}"

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, scheduler=None, **kwargs):
        bsz = q_tokens.size(0)

        # Query encoder를 통해 Query의 feature를 추출
        # self.tokenizer.batch_decode(q_tokens, skip_special_tokens=True) # 2018
        # self.tokenizer.batch_decode(kwargs['wp_tokens'], skip_special_tokens=True) # 2023
        # self.tokenizer.batch_decode(kwargs['p_tokens'], skip_special_tokens=True) # 2018
        q_q = self.encoder_q(input_ids=q_tokens,
                             attention_mask=q_mask, normalize=self.norm_query)
        q_p = self.encoder_q(
            input_ids=kwargs['p_tokens'], attention_mask=kwargs['p_mask'], normalize=self.norm_doc)
        q_wp = self.encoder_q(
            input_ids=kwargs['wp_tokens'], attention_mask=kwargs['wp_mask'], normalize=self.norm_doc)

        # Reference model
        with torch.no_grad():  # no gradient to keys
            if stats_prefix == "train":
                # update the key encoder
                self._momentum_update_k_encoder()  # update the key encoder
            elif stats_prefix == "dev" or stats_prefix == "test":
                self.encoder_k.eval()

            if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                self.encoder_k.eval()

            # 같은 K encoder를 사용하여 P(Positive), WP(Weak Positive)의 feature를 추출
            k_q = self.encoder_k(
                input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
            k_p = self.encoder_k(
                input_ids=kwargs['p_tokens'], attention_mask=kwargs['p_mask'], normalize=self.norm_doc)
            k_wp = self.encoder_k(
                input_ids=kwargs['wp_tokens'], attention_mask=kwargs['wp_mask'], normalize=self.norm_doc)

            # ref_p = self.encoder_ref(input_ids=kwargs['p_tokens'], attention_mask=kwargs['p_mask'], normalize=self.norm_doc)
            # ref_wp = self.encoder_ref(input_ids=kwargs['wp_tokens'], attention_mask=kwargs['wp_mask'], normalize=self.norm_doc)

        kl_loss = torch.nn.functional.kl_div(F.log_softmax(
            q_q, dim=-1), F.softmax(k_p, dim=-1), reduction="batchmean")

        q_aligned_scores = torch.einsum(
            "id, id->i", q_q / self.temperature, q_p)
        q_unaligned_scores = torch.einsum(
            "id, id->i", q_q / self.temperature, q_wp)

        k_aligned_scores = torch.einsum(
            "id, id->i", k_q / self.temperature, k_p)
        k_unaligned_scores = torch.einsum(
            "id, id->i", k_q / self.temperature, k_wp)

        '''
        q_aligned_scores = torch.einsum("id, jd->ij", q_p / self.temperature, q_p)
        q_unaligned_scores = torch.einsum("id, jd->ij", q_wp / self.temperature, q_wp)

        k_aligned_scores = torch.einsum("id, jd->ij", k_p / self.temperature, k_p)
        k_unaligned_scores = torch.einsum("id, jd->ij", k_wp / self.temperature, k_wp)
        '''
        # q encoder logratio
        q_sub = q_aligned_scores - q_unaligned_scores

        # calculate q_sub_loss
        # we give 1e-5 (a small value) to let model favor the q_aligned_scores
        # directly minimize the q_sub log ratio
        # 1) to prefer q_aligned_scores over q_unaligned_scores
        # 2) to make q_sub close to 0 (q_aligned_scores and q_unaligned_scores are close)
        # q_sub_loss = torch.nn.functional.relu(-q_sub)

        # ref encoder logratio
        p_sub = k_aligned_scores - k_unaligned_scores

        # final logits is the difference between the log-ratio of the query encoder and the reference encoder (just like DPO)
        logits = q_sub - p_sub

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        rank_loss = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        # get mean of the losses # Ranking loss는 그 의의 상으로는 그냥 전체에 대한 ranking loss를 구하는 것이 맞을 수 있지 않나? (먹고 생각해보기)
        rank_loss = rank_loss  # + q_sub_loss

        # rank_loss_1 = torch.nn.functional.margin_ranking_loss(q_sub, p_sub, target=torch.ones(size=q_sub.shape).cuda(), margin=1.0, reduction='mean')
        # q_q, k_p를 예측하게 만드는게 학습을 어렵게 하면서 더 좋게 만들어줄 수 있지 않을까?
        l_pos = torch.einsum("nc,kc->nk", [q_q, k_p])

        # concat queue and q_wp (This will be hard examples - related but time shifted)
        # l_neg = torch.einsum("nc,ck->nk", [q_q, torch.cat([self.queue.clone().detach(), q_wp.T], dim=-1)])
        # l_neg = torch.einsum("nc,ck->nk", [q_q, self.queue.clone().detach()])
        l_neg = torch.einsum("nc,ck->nk", [q_q, self.queue.clone().detach()])
        l_weak_pos = torch.einsum("nc,kc->nk", [q_q, k_wp])

        # remove diagonal and shrink (hard negatives == number of batch size - 1)
        # 정답을 제외한 나머지를 hard negative로 사용 (내일 꼭 사용.)
        l_hard_neg = l_weak_pos[~torch.eye(l_weak_pos.shape[0], dtype=bool)].view(
            l_weak_pos.shape[0], -1)
        scores = torch.cat([l_pos, l_neg, l_hard_neg], dim=1)

        # scores = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)
        e_loss = torch.nn.functional.cross_entropy(
            scores, labels, label_smoothing=self.label_smoothing)
        predicted_idx = torch.argmax(scores, dim=-1)

        # rank_loss = ((predicted_idx != labels) * rank_loss).mean()
        rank_loss = rank_loss.mean()

        alpha = 0.95  # rank_loss / (e_loss + rank_loss)
        # clip rank_loss to max 1.0
        # rank_loss = torch.clamp(rank_loss, min=0.0, max=1.0)
        loss = alpha * (e_loss) + (1 - alpha) * (rank_loss)
        # loss = alpha * (e_loss + kl_loss) + (1 - alpha) * (rank_loss)

        self._dequeue_and_enqueue(torch.cat([k_p, k_wp], dim=0))
        # self._dequeue_and_enqueue(k_p)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"

        iter_stats[f"{stats_prefix}rank_loss"] = (rank_loss.item(), bsz)
        iter_stats[f"{stats_prefix}e_loss"] = (e_loss.item(), bsz)
        iter_stats[f"{stats_prefix}kl_loss"] = (kl_loss.item(), bsz)
        iter_stats[f"{stats_prefix}all_loss"] = (loss.item(), bsz)

        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(q_q, dim=0).mean().item()
        stdp = torch.std(q_p, dim=0).mean().item()
        stdwp = torch.std(q_wp, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdp"] = (stdp, bsz)
        iter_stats[f"{stats_prefix}stdwp"] = (stdwp, bsz)

        return loss, iter_stats
