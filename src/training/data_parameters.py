# From https://github.com/apple/ml-data-parameters
# Copyright (C) 2019 Apple Inc. All Rights Reserved. See
# https://github.com/apple/ml-data-parameters/blob/master/LICENSE

import numpy as np
import torch

def get_class_inst_data_params_n_optimizer(nr_classes, nr_instances, device):
    """Returns class and instance level data parameters and their corresponding optimizers.
    Args:
        nr_classes (int):  number of classes in dataset.
        nr_instances (int): number of instances in dataset.
        device (str): device on which data parameters should be placed.
    Returns:
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters
        optimizer_class_param (SparseSGD): Sparse SGD optimizer for class parameters
        optimizer_inst_param (SparseSGD): Sparse SGD optimizer for instance parameters
    """

    class_parameters = torch.tensor(
        np.ones(nr_classes) * np.log(1.0),
        dtype=torch.float32,
        requires_grad=True,
        device=device
    )
    optimizer_class_param = SparseSGD(
        [class_parameters],
        lr=0.1,
        momentum=0.9,
        skip_update_zero_grad=True
    )

    inst_parameters = torch.tensor(
        np.ones(nr_instances) * np.log(1.0),
        dtype=torch.float32,
        requires_grad=True,
        device=device
    )
    optimizer_inst_param = SparseSGD(
        [inst_parameters],
        lr=0.1,
        momentum=0.9,
        skip_update_zero_grad=True
    )

    return class_parameters, inst_parameters, optimizer_class_param, optimizer_inst_param

class SparseSGD(torch.optim.SGD):
    """
    This class implements SGD for optimizing parameters where at each iteration only few parameters obtain a gradient.
    More specifically, we zero out the update to state and momentum buffer for parameters with zero gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        skip_update_zero_grad (bool, optional): if True, we will zero out the update to state and momentum buffer
                                                for parameters which are not in computation graph (eq. to zero gradient).
    """

    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, skip_update_zero_grad=False):
        super(SparseSGD, self).__init__(params,
                                        lr=lr,
                                        momentum=momentum,
                                        dampening=dampening,
                                        weight_decay=weight_decay,
                                        nesterov=nesterov)

        self.skip_update_zero_grad = skip_update_zero_grad
        assert weight_decay == 0, 'Weight decay for optimizer should be set to 0. ' \
                                  'For data parameters, we explicitly invoke weight decay on ' \
                                  'subset of data parameters in the computation graph.'

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Generating pointers to old-state
                p_before_update = p.data.clone()

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # Initializes momentum buffer
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                        buf_before_update = None
                    else:
                        buf = param_state['momentum_buffer']
                        buf_before_update = buf.data.clone()
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

                # We need to revert back the state of parameter and momentum buffer for entries with zero-grad
                if self.skip_update_zero_grad:
                    indices_without_grad = torch.abs(p.grad) == 0.0

                    # Old Momentum buffer has updated parameters without gradient, reverting to old value
                    p.data[indices_without_grad] = p_before_update.data[indices_without_grad]

                    # Resetting momentum buffer parameters without gradient
                    if (buf_before_update is not None) and (momentum != 0):
                        param_state['momentum_buffer'].data[indices_without_grad] = \
                            buf_before_update.data[indices_without_grad]
        return loss
