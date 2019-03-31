# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

import visdom
vis = visdom.Visdom()

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:

            print('Metrics Dictionary =======')
            temp_dict = {
                'loss': [],
                'loss_box_reg': [],
                'loss_classifier': [],
                'loss_mask': [],
                'loss_objectness': [],
                'loss_rpn_box_reg': [],
                'time': []
            }

            temp_dict['loss'].append(str(meters).split('  ')[0].split(':')[1].split(' ')[0])
            temp_dict['loss_box_reg'].append(str(meters).split('  ')[1].split(':')[1].split(' ')[0])
            temp_dict['loss_classifier'].append(str(meters).split('  ')[2].split(':')[1].split(' ')[0])
            temp_dict['loss_mask'].append(str(meters).split('  ')[3].split(':')[1].split(' ')[0])
            temp_dict['loss_objectness'].append(str(meters).split('  ')[4].split(':')[1].split(' ')[0])
            temp_dict['loss_rpn_box_reg'].append(str(meters).split('  ')[5].split(':')[1].split(' ')[0])
            temp_dict['time'].append(str(meters).split('  ')[6].split(':')[1].split(' ')[0])

            print(temp_dict)
            # # loss
            losstrace = dict(x=temp_dict['time'], y=temp_dict['loss'], mode="markers+lines", type='custom',
                         marker={'color': 'red', 'symbol': 104, 'size': "10"},
                        name='1st Trace')
            losslayout = dict(title="loss", xaxis={'title': 'time'}, yaxis={'title': 'loss'})
            vis._send({'data': [losstrace], 'layout': losslayout, 'win': 'mywinloss'})

            # # loss_box_reg
            loss_box_regtrace = dict(x=temp_dict['time'], y=temp_dict['loss_box_reg'], mode="markers+lines", type='custom',
                             marker={'color': 'red', 'symbol': 104, 'size': "10"},
                             name='2nd Trace')
            loss_box_reglayout = dict(title="loss_box_reg", xaxis={'title': 'time'}, yaxis={'title': 'loss_box_reg'})
            vis._send({'data': [loss_box_regtrace], 'layout': loss_box_reglayout, 'win': 'mywinloss_box_reg'})

            # # loss_classifier
            loss_classifiertrace = dict(x=temp_dict['time'], y=temp_dict['loss_classifier'], mode="markers+lines",
                                     type='custom',
                                     marker={'color': 'red', 'symbol': 104, 'size': "10"},
                                     name='3rd Trace')
            loss_classifierlayout = dict(title="loss_classifier", xaxis={'title': 'time'}, yaxis={'title': 'loss_classifier'})
            vis._send({'data': [loss_classifiertrace], 'layout': loss_classifierlayout, 'win': 'mywinloss_classifier'})

            # # loss_mask
            loss_masktrace = dict(x=temp_dict['time'], y=temp_dict['loss_mask'], mode="markers+lines",
                                        type='custom',
                                        marker={'color': 'red', 'symbol': 104, 'size': "10"},
                                        name='4th Trace')
            loss_masklayout = dict(title="loss_mask", xaxis={'title': 'time'}, yaxis={'title': 'loss_mask'})
            vis._send({'data': [loss_masktrace], 'layout': loss_masklayout, 'win': 'mywinloss_mask'})

            # # loss_objectness
            loss_objectnesstrace = dict(x=temp_dict['time'], y=temp_dict['loss_objectness'], mode="markers+lines",
                                        type='custom',
                                        marker={'color': 'red', 'symbol': 104, 'size': "10"},
                                        name='5th Trace')
            loss_objectnesslayout = dict(title="loss_objectness", xaxis={'title': 'time'}, yaxis={'title': 'loss_objectness'})
            vis._send({'data': [loss_objectnesstrace], 'layout': loss_objectnesslayout, 'win': 'mywinloss_objectness'})

            # # loss_rpn_box_reg
            loss_rpn_box_regtrace = dict(x=temp_dict['time'], y=temp_dict['loss_rpn_box_reg'], mode="markers+lines",
                                        type='custom',
                                        marker={'color': 'red', 'symbol': 104, 'size': "10"},
                                        name='6th Trace')
            loss_rpn_box_reglayout = dict(title="loss_rpn_box_reg", xaxis={'title': 'time'},
                                         yaxis={'title': 'loss_rpn_box_reg'})
            vis._send({'data': [loss_rpn_box_regtrace], 'layout': loss_rpn_box_reglayout, 'win': 'mywinloss_rpn_box_reg'})

            # #
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
