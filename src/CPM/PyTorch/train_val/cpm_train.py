# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import AverageMeter as AverageMeter
from utils.utils import save_checkpoint as save_checkpoint
from utils.utils import Config as Config
import cpm_model
import lsp_lspet_data
import Mytransforms


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default='../ckpt/cpm_latest.pth.tar',type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--model_name', default='../ckpt/cpm', type=str,
                        help='model name to save parameters')

    return parser.parse_args()


def construct_model(args):

    model = cpm_model.CPM(k=14)
    # load pretrained model
    # state_dict = torch.load(args.pretrained)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    return model

def get_parameters(model, config, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr},
            {'params': lr_2, 'lr': config.base_lr * 2.},
            {'params': lr_4, 'lr': config.base_lr * 4.},
            {'params': lr_8, 'lr': config.base_lr * 8.}]

    return params, [1., 2., 4., 8.]



def train_val(model, args):

    train_dir = args.train_dir
    val_dir = args.val_dir

    config = Config(args.config)
    cudnn.benchmark = True

    # train
    train_loader = torch.utils.data.DataLoader(
        lsp_lspet_data.LSP_Data('lspet', train_dir, 8,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(368),
                Mytransforms.RandomHorizontalFlip(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)
    # val
    if args.val_dir is not None and config.test_interval != 0:
        # val
        val_loader = torch.utils.data.DataLoader(
            lsp_lspet_data.LSP_Data('lsp', val_dir, 8,
                              Mytransforms.Compose([Mytransforms.TestResized(368),
                                                    ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()

    params, multiple = get_parameters(model, config, False)

    optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(6)]
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model

    heat_weight = 46 * 46 * 15 / 1.0

    while iters < config.max_iter:

        for i, (input, heatmap, centermap) in enumerate(train_loader):

            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy,
                                                 policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            heatmap = heatmap.cuda(async=True)
            centermap = centermap.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)
            centermap_var = torch.autograd.Variable(centermap)


            heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

            loss1 = criterion(heat1, heatmap_var) * heat_weight
            loss2 = criterion(heat2, heatmap_var) * heat_weight
            loss3 = criterion(heat3, heatmap_var) * heat_weight
            loss4 = criterion(heat4, heatmap_var) * heat_weight
            loss5 = criterion(heat5, heatmap_var) * heat_weight
            loss6 = criterion(heat6, heatmap_var) * heat_weight


            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            losses.update(loss.data[0], input.size(0))
            for cnt, l in enumerate(
                    [loss1, loss2, loss3, loss4, loss5, loss6]):
                losses_list[cnt].update(l.data[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                      'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                      'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                      'Learning rate = {2}\n'
                      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, config.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0, 6):
                    print('Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                          .format(cnt + 1, loss1=losses_list[cnt]))

                print time.strftime(
                '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',time.localtime())

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(6):
                    losses_list[cnt].reset()

            save_checkpoint({
                'iter': iters,
                'state_dict': model.state_dict(),
            }, 0, args.model_name)

            # val
            if args.val_dir is not None and config.test_interval != 0 and iters % config.test_interval == 0:

                model.eval()
                for j, (input, heatmap, centermap) in enumerate(val_loader):
                    heatmap = heatmap.cuda(async=True)
                    centermap = centermap.cuda(async=True)

                    input_var = torch.autograd.Variable(input)
                    heatmap_var = torch.autograd.Variable(heatmap)
                    centermap_var = torch.autograd.Variable(centermap)

                    heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

                    loss1 = criterion(heat1, heatmap_var) * heat_weight
                    loss2 = criterion(heat2, heatmap_var) * heat_weight
                    loss3 = criterion(heat3, heatmap_var) * heat_weight
                    loss4 = criterion(heat4, heatmap_var) * heat_weight
                    loss5 = criterion(heat5, heatmap_var) * heat_weight
                    loss6 = criterion(heat6, heatmap_var) * heat_weight

                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                    losses.update(loss.data[0], input.size(0))
                    for cnt, l in enumerate(
                            [loss1, loss2, loss3, loss4, loss5, loss6]):
                        losses_list[cnt].update(l.data[0], input.size(0))

                    batch_time.update(time.time() - end)
                    end = time.time()
                    is_best = losses.avg < best_model
                    best_model = min(best_model, losses.avg)
                    save_checkpoint({
                        'iter': iters,
                        'state_dict': model.state_dict(),
                    }, is_best, args.model_name)

                    if j % config.display == 0:
                        print('Test Iteration: {0}\t'
                              'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                              'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                              'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                            j, config.display, batch_time=batch_time,
                            data_time=data_time, loss=losses))
                        for cnt in range(0, 6):
                            print('Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                                  .format(cnt + 1, loss1=losses_list[cnt]))

                        print time.strftime(
                            '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
                            time.localtime())
                        batch_time.reset()
                        losses.reset()
                        for cnt in range(6):
                            losses_list[cnt].reset()

                model.train()


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse()
    model = construct_model(args)
    train_val(model, args)
