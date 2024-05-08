import logging
import math
from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class ParamScheduler:

    def __init__(
        self,
        epochs=20000,
        # lr_scheduler=None,
        # betas_scheduler=None,
        # group_weights_scheduler=None,
        default_lr=1e-3,
        default_betas=(0.99, 0.99),
        default_group_weights=(0.5, 0.5),
    ):
        self.max_epochs = epochs
        self.epochs = 0
        # self.lr_scheduler = lr_scheduler
        # self.betas_scheduler = betas_scheduler
        # self.group_weights_scheduler = group_weights_scheduler
        self.default_lr = default_lr
        self.default_betas = default_betas
        self.default_group_weights = default_group_weights
        

    def lr(self):
        # if self.lr_scheduler is not None:
        #     return self.lr_scheduler(self.epochs, self.max_epochs, self.grouped_losses)
        return self.default_lr

    def betas(self):
        # if self.betas_scheduler is not None:
        #     return self.betas_scheduler(self.epochs, self.max_epochs, self.grouped_losses)
        return self.default_betas

    def group_weights(self):
        # if self.group_weights_scheduler is not None:
        #     return torch.tensor(self.group_weights_scheduler(self.epochs, self.max_epochs, self.grouped_losses))
        return self.default_group_weights

    def step(self, losses, grouped_losses):
        self.epochs += 1
        self.losses = losses
        self.grouped_losses = grouped_losses


def sadam(params, grads, exp_avgs, exp_avg_sqs,state_steps, beta1, beta2, lr, eps,  group_weights):


    r"""Functional API that performs MultiAdam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    # n_group is num of different group_weights
    # n_params is the number of all params
    n_groups, n_params = len(grads), len(grads[0])
    grads_cat, exp_avgs_cat, exp_avg_sqs_cat= [], [], []

    for i in range(n_params):
        grads_cat.append(torch.stack([grads[j][i] for j in range(n_groups)]))
        exp_avgs_cat.append(torch.stack([exp_avgs[j][i] for j in range(n_groups)]))
        exp_avg_sqs_cat.append(torch.stack([exp_avg_sqs[j][i] for j in range(n_groups)]))


    for i, param in enumerate(params):

        grad = -grads_cat[i]  # torch.stack([p.grad for different losses])
        exp_avg = exp_avgs_cat[i]
        exp_avg_sq = exp_avg_sqs_cat[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step



        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        update_raw = exp_avg / denom  # raw update for every loss group
        update = (update_raw * group_weights.view((-1, ) + (1, ) * (exp_avg.dim() - 1))).sum(dim=0)  # weighted sum for current param

      

        param -= step_size * update

    # update states
    for i in range(n_groups):
        for j in range(n_params):
            exp_avgs[i][j].copy_(exp_avgs_cat[j][i])
            exp_avg_sqs[i][j].copy_(exp_avg_sqs_cat[j][i])



class MultiAdam(Optimizer):

    def __init__(self,params,lr=1e-3,betas=(0.99, 0.99),eps=1e-8,loss_group_idx=None,):

        # agg_betas = (0, 0)
        self.is_init_state = True
        self.loss_group_idx = loss_group_idx
        self.n_groups = len(self.loss_group_idx) + 1
        self.group_weights = 1 / self.n_groups * torch.ones([self.n_groups])        
        param_scheduler = ParamScheduler(default_lr=lr, default_betas=betas, default_group_weights=self.group_weights)
        self.param_scheduler = param_scheduler

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            # agg_betas=agg_betas,
        )
        super(MultiAdam, self).__init__(params, defaults) # hello

    # def __setstate__(self, state):
    #     print("hello")
    #     print(state)
    #     super(MultiAdam, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)
        #     group.setdefault('maximize', False)
        #     group.setdefault('agg_momentum', False)

    def init_states(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(self.n_groups)]
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(self.n_groups)]
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(self.n_groups)]
                state['agg_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['agg_exp_avg_sqs'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        self.is_init_state = False

    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        with torch.enable_grad():
            _ = closure(skip_backward=True)
            losses = self.losses

            loss_group_idx = [0] + self.loss_group_idx + [len(losses)]
            grouped_losses = []
            for i in range(len(loss_group_idx) - 1):
                grouped_losses.append(torch.sum(losses[loss_group_idx[i]:loss_group_idx[i + 1]]))

        assert len(grouped_losses) == self.n_groups
        self.zero_grad()
        self.param_scheduler.step(losses=self.losses, grouped_losses=grouped_losses)

        params_with_grad = []
        grads_groups = []
        exp_avgs_groups = []
        exp_avg_sqs_groups = []
        # max_exp_avg_sqs_groups = []

        # agg_exp_avg = []
        # agg_exp_avg_sqs = []

        if self.is_init_state:
            self.init_states()

        for i, loss in enumerate(grouped_losses):
            loss.backward(retain_graph=True)

            for group in self.param_groups:
                grads = []
                exp_avgs = []
                exp_avg_sqs = []


                # update loss specific parameters: p.grad, exp_avg, exp_avg_sq, max_exp_avg_sq
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad.clone())
                        p.grad.zero_()

                        state = self.state[p]

                        exp_avgs.append(state['exp_avg'][i])
                        exp_avg_sqs.append(state['exp_avg_sq'][i])





                grads_groups.append(grads)
                exp_avgs_groups.append(exp_avgs)
                exp_avg_sqs_groups.append(exp_avg_sqs)


        with torch.no_grad():
            temp=self.param_groups
            for group in self.param_groups:
                params_with_grad = []
                state_steps = []
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        # update the steps for each param group update
                        self.state[p]['step'] += 1
                        state_steps.append(self.state[p]['step'])

                beta1, beta2 = self.param_scheduler.betas()
                
                sadam(
                    params=params_with_grad,  # list of params(which has grad)
                    # list[list[Tensor]]: dim0 is different loss_group,
                    # dim1 is grads of every params for different losses
                    grads=grads_groups,
                    exp_avgs=exp_avgs_groups,
                    exp_avg_sqs=exp_avg_sqs_groups,
                    state_steps=state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=self.param_scheduler.lr(),
                    eps=group['eps'],

                    group_weights=self.param_scheduler.group_weights(),

                )

        return grouped_losses