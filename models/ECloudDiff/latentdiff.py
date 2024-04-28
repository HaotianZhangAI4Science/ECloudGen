import torch
import math
import os

import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from models import register_model
from models.ECloudDiff.unet import UNetModel
from models.ECloudDiff.encoder import Pocket3DCNNEncoder
from models.ECloudDiff.encoder import Ligand3DCNNEncoder
from models.ECloudDiff.decoder import Ecloud3DCNNDecoder
from models.ECloudDiff.quantize import VectorQuantizer, GumbelQuantizer
from torch import einsum
from abc import ABC, abstractmethod
from models.ECloudDiff.nn import mean_flat

@register_model(['eclouddiff'])
class EcloudLatentDiffusionModel(nn.Module):
    def __init__(self, cfg):
        super(EcloudLatentDiffusionModel, self).__init__()
        self.cfg = cfg
        self.PocketEncoder = Pocket3DCNNEncoder(cfg)
        self.LigandEncoder = Ligand3DCNNEncoder(cfg)
        self.EcloudDecoder = Ecloud3DCNNDecoder(cfg)

        if self.cfg.MODEL.FREEZE_ENCODER_DECODER:
            self.PocketEncoder = self.PocketEncoder.eval()
            self.LigandEncoder = self.LigandEncoder.eval()
            self.EcloudDecoder = self.EcloudDecoder.eval()

        if not self.cfg.MODEL.FREEZE_DIFFUSION_MODEL:
            self.model = UNetModel(**cfg.MODEL.DIFFUSION.UNET)

        if self.cfg.MODEL.DIFFUSION.add_vqvae_loss:
            embed_dim = self.cfg.MODEL.PKT_ENCODER.OUTPUT_DIM + self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM
            # self.quantizer = VectorQuantizer(embed_dim=embed_dim, n_embed=8192)
            self.quantizer = GumbelQuantizer(num_hiddens=embed_dim, embedding_dim=embed_dim, n_embed=8192)

        # ddpm
        betas = get_named_beta_schedule('linear', self.cfg.MODEL.DIFFUSION.time_step)
        self.model_mean_type = self.cfg.MODEL.DIFFUSION.model_mean_type
        self.rescale_timesteps = self.cfg.MODEL.DIFFUSION.rescale_timesteps
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg)

    def training_losses(self, pkt, lig, t, return_logit, is_condition=False):
        bs, size = lig.shape[:2]
        pkt_z = self.PocketEncoder(pkt)
        lig_z = self.LigandEncoder(lig)
        if self.cfg.MODEL.DIFFUSION.add_vqvae_loss:
            lig_z, vq_loss, encoding_indices = self.quantizer(lig_z)
        terms = {}
        if not self.cfg.MODEL.FREEZE_DIFFUSION_MODEL:
            x_start_mean = lig_z.permute(0, 4, 1, 2, 3).contiguous()
            _pkt_z = pkt_z.permute(0, 4, 1, 2, 3).contiguous()
            # x_start_mean = z
            std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                       torch.tensor([0]).to(x_start_mean.device),
                                       x_start_mean.shape)
            x_start_log_var = 2 * torch.log(std)
            # This step can be regarded as a first-order Markov process
            x_start = self.get_x_start(x_start_mean, std)
            # x_start = x_start_mean
            # p(xt|x0)
            noise = torch.randn_like(x_start)
            x_t_lig = self.q_sample(x_start, t, noise=noise)  # reparametrization trick.
            x_t = torch.cat([_pkt_z, x_t_lig], dim=1)
            model_output = self.model(x_t, self._scale_timesteps(t))
            # 3 modeling methods, choose different tags
            target = {
                'PREVIOUS_X': self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t_lig, t=t
                )[0],
                'X0': x_start,
                'EPS': noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)

            # Encoder loss
            model_out_x_start = self.x0_helper(model_output, x_t_lig, t)['pred_xstart']
            t0_mask = (t == 0)
            t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
            terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])
            terms["mse"] = terms["mse"].mean()
            """
            Since the Encoder has parameters, p(xt| x0) = p_gamma(xt| x0), gamma is the encoder parameter, so this loss needs to be added
            In DDPM, this loss does not have parameters, so it is a constant and can be ignored, but here there is an Encoder, this is no longer a constant, and the loss needs to be calculated separately
            From the form of loss, it can be seen that the goal is to be a regular term for the Encoder output to prevent the Encoder output value from being too large
            """
            if not self.cfg.MODEL.FREEZE_ENCODER_DECODER and not self.cfg.MODEL.FREEZE_DIFFUSION_MODEL:
                out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
                tT_loss = mean_flat(out_mean ** 2)
                terms['tT_loss'] = tT_loss.mean()
            else:
                terms['tT_loss'] = torch.tensor(0.0)

            terms['posterior_loss'] = terms["mse"].mean()



        # Decoder loss
        if self.cfg.MODEL.DIFFUSION.add_noise:
            lig_z = lig_z + 0.2 * (torch.var(lig_z, dim=-1, keepdim=True) ** 0.5) * torch.randn_like(lig_z)

        z = torch.cat([pkt_z, lig_z], dim=-1)

        logits = self.EcloudDecoder(z).squeeze(1)
        lig_ecloud = lig.type(logits.dtype)
        decoder_loss = nn.MSELoss(reduction='mean')(logits, lig_ecloud)
        contra_loss = torch.sum(lig_ecloud * pkt) / (pkt.size(0)*pkt.size(1)*pkt.size(2)*pkt.size(3))
        terms['decoder_loss'] = decoder_loss
        terms['contra_loss'] = contra_loss

        if self.cfg.MODEL.FREEZE_DIFFUSION_MODEL:
            terms["loss"] = decoder_loss + contra_loss
        else:
            terms["loss"] = 10*terms["mse"].mean() + terms['tT_loss'] + decoder_loss + contra_loss

        if self.cfg.MODEL.DIFFUSION.add_vqvae_loss:
            terms['vq_loss'] = vq_loss
            terms["loss"] = terms["loss"] + 10 * vq_loss

        if return_logit:
            terms['logits'] = logits
        return terms


    def x0_helper(self, model_output, x, t):
        if self.model_mean_type == 'PREVIOUS_X':
            pred_xstart =  self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            pred_prev = model_output

        elif self.model_mean_type in ['X0', 'EPS']:
            if self.model_mean_type == 'X0':
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:
            raise NotImplementedError(self.model_mean_type)
        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_x_start(self, x_start_mean, std):
        '''
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        '''
        noise = torch.randn_like(x_start_mean)
        # print(std.shape, noise.shape, x_start_mean.shape)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def forward(self, pkt, lig, net_output=None, return_logit=False, is_condition=False):
        t = torch.randint(0, self.num_timesteps, (lig.shape[0],), device=lig.device).long()
        return self.training_losses(pkt, lig, t, return_logit, is_condition=is_condition)

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(1)
        assert t.shape == (B,)

        model_output = model.model(x, self._scale_timesteps(t), **model_kwargs)

        # model_variance, model_log_variance = np.append(self.posterior_variance[1], self.betas[1:]),\
        #                                      np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_variance, model_log_variance = self.posterior_variance, self.posterior_log_variance_clipped

        model_variance = _extract_into_tensor(model_variance, t, x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...].shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...].shape)

        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == 'PREVIOUS_X':
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...], t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in ['X0', 'EPS']:
            if self.model_mean_type == 'X0':
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...], t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...], t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...].shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = torch.randn_like(x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...])
            replace_mask = torch.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = torch.randn_like(noise[replace_mask])
                replace_mask = torch.abs(noise) > top_p
            assert (torch.abs(noise) <= top_p).all()

        else:
            noise = torch.randn_like(x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x[:, self.cfg.MODEL.LIG_ENCODER.OUTPUT_DIM:, ...].shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out}

    def p_sample_loop(
        self,
        model,
        shape,
        condition=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            condition=condition,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        condition=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            if condition is not None:
                img = torch.cat([condition, img], dim=1)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                # if i < 600:
                #     a = 0
                img = out["sample"]


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)