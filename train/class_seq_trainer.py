import os
import time
import shutil
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel

from data.test_data_loader import DeepSpeakerTestDataset, DeepSpeakerTestDataLoader
from data.utt_seq_data_loader import DeepSpeakerUttSeqDataset, DeepSpeakerUttSeqDataLoader
from data.data_sampler import BucketingSampler, DistributedBucketingSampler
from model.model import model_select, load
from model.margin import margin_select
from train.base_trainer import reduce_loss_dict, evaluate
from utils import utils


def train(opt, logging):    
    ## Data Prepare ## 
    if opt.main_proc:
        logging.info("Building dataset")
    
    train_dataset = DeepSpeakerUttSeqDataset(opt, os.path.join(opt.dataroot, 'train_combined_no_sil'))
    if not opt.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=opt.batch_size,
                                                    num_replicas=opt.num_gpus, rank=opt.local_rank)
    train_loader = DeepSpeakerUttSeqDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
                                           
    val_dataset = DeepSpeakerTestDataset(opt, os.path.join(opt.dataroot, 'voxceleb1_test_no_sil'))
    val_loader = DeepSpeakerTestDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False, pin_memory=True)
    
    opt.in_size = train_dataset.in_size
    opt.out_size = train_dataset.class_nums  
    print('opt.in_size {} opt.out_size {}'.format(opt.in_size, opt.out_size))  
                                           
    if opt.main_proc:
        logging.info("Building dataset Sucessed")
    
    ##  Building Model ##
    model = model_select(opt, seq_training=True)
    margin = margin_select(opt)
    
    if opt.resume:
        model, opt.total_iters = load(model, opt.resume, 'state_dict')
        margin, opt.total_iters = load(margin, opt.resume, 'margin_state_dict')
    
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    if opt.optim_type == 'sgd':
        optimizer = optim.SGD([
            {'params': model.parameters(), 'weight_decay': 5e-4},
            {'params': margin.parameters(), 'weight_decay': 5e-4}
        ], lr=opt.lr, momentum=0.9, nesterov=True)
    elif opt.optim_type == 'adam':
        optimizer = optim.Adam([
            {'params': model.parameters(), 'weight_decay': 5e-4},
            {'params': margin.parameters(), 'weight_decay': 5e-4}
        ], lr=opt.lr, betas=(opt.beta1, 0.999))
        
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8, 14, 20], gamma=0.1)
        
    model.to(opt.device)
    margin.to(opt.device)
    
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank)
        margin = torch.nn.parallel.DistributedDataParallel(margin, device_ids=[opt.local_rank],
                                                           output_device=opt.local_rank)
    if opt.main_proc:
        print(model)
        print(margin)
        logging.info("Building Model Sucessed") 
        
    best_perform_eer = 1.0
    
    losses = utils.AverageMeter()
    class_losses = utils.AverageMeter()
    segment_class_losses = utils.AverageMeter()
    penalty_losses = utils.AverageMeter()
    acc = utils.AverageMeter()
    
    save_model = model
    if isinstance(model, DistributedDataParallel):
        save_model = model.module
        
    # Initial performance
    '''if opt.main_proc:
        EER = evaluate(opt, model, val_loader, logging)
        best_perform_eer = EER
        print('>>Start performance: EER = {}<<'.format(best_perform_eer))'''
    
    total_iters = opt.total_iters
    for epoch in range(1, opt.total_epoch + 1):
        train_sampler.shuffle(epoch)
        scheduler.step()
        # train model
        if opt.main_proc:
            logging.info('Train Epoch: {}/{} ...'.format(epoch, opt.total_epoch))
        model.train()
        margin.train()

        since = time.time()
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, inputs, segment_nums, targets, segment_targets = data
            ##print(inputs.shape, segment_nums)
            inputs = inputs.to(opt.device)
            segment_nums = segment_nums.to(opt.device)
            targets = targets.to(opt.device)
            segment_targets = segment_targets.to(opt.device)
            optimizer.zero_grad()
            
            logits, segment_logits, attn, w, b = model(inputs, segment_nums)
            outputs = margin(logits, targets)
            class_loss = criterion(outputs, targets)
            
            segment_outputs = margin(segment_logits, targets)
            segment_class_loss = criterion(segment_outputs, segment_targets) * opt.segment_loss_lamda
        
            if opt.att_type == 'multi_attention' and attn is not None:
                penalty_loss = opt.penalty_loss_lamda * save_model.penalty_loss_cal(attn)
            else:
                penalty_loss = 0
            
            loss_dict_reduced = reduce_loss_dict(opt, {'class_loss': class_loss, 'penalty_loss': penalty_loss, 
                                                       'segment_class_loss': segment_class_loss})                
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            class_loss_value = loss_dict_reduced['class_loss'].item()
            penalty_loss_value = loss_dict_reduced['penalty_loss'].item()
            segment_class_loss_value = loss_dict_reduced['segment_class_loss'].item()
            loss = class_loss + penalty_loss + segment_class_loss
                
            # Check the loss and avoid the invaided loss
            inf = float("inf")
            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
                continue
                    
            loss.backward()
            if utils.check_grad(model.parameters(), opt.clip_grad, opt.ignore_grad):
                if opt.main_proc:
                    logging.info('Not a finite gradient or too big, ignoring')
                optimizer.zero_grad()
                continue
            optimizer.step()

            total_iters += opt.num_gpus
            # Update the loss for logging
            losses.update(loss_value)
            class_losses.update(class_loss_value)
            penalty_losses.update(penalty_loss_value)
            segment_class_losses.update(segment_class_loss_value)
            
            # print train information
            if total_iters % opt.print_freq == 0 and opt.main_proc:
                # current training accuracy
                _, predict = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (np.array(predict.cpu()) == np.array(targets.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                logging.info("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f} ({:.4f}) [ class: {:.4f}, segment_class: {:.4f}, penalty_loss {:.4f}], train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, loss_value, losses.avg, class_losses.avg, segment_class_losses.avg, penalty_losses.avg, correct/total, time_cur, scheduler.get_lr()[0]))
              
            # save model
            if total_iters % opt.save_freq == 0 and opt.main_proc:
                logging.info('Saving checkpoint: {}'.format(total_iters))
                if opt.distributed:
                    model_state_dict = model.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    margin_state_dict = margin.state_dict()
                state = {'state_dict': model_state_dict, 'margin_state_dict': margin_state_dict, 'total_iters': total_iters,}
                filename = 'newest_model.pth'
                if os.path.isfile(os.path.join(opt.model_dir, filename)):
                    shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'newest_model.pth_bak'))
                utils.save_checkpoint(state, opt.model_dir, filename=filename)
                    
            # Validate the trained model
            if total_iters % opt.validate_freq == 0:
                EER = evaluate(opt, model, val_loader, logging)
                ##scheduler.step(EER)
                
                if opt.main_proc and EER < best_perform_eer:
                    best_perform_eer = EER
                    logging.info("Found better validated model (EER = %.3f), saving to model_best.pth" % (best_perform_eer))
                    if opt.distributed:
                        model_state_dict = model.module.state_dict()
                        margin_state_dict = margin.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()
                        margin_state_dict = margin.state_dict()
                    state = {'state_dict': model_state_dict, 'margin_state_dict': margin_state_dict, 'total_iters': total_iters,}  
                    filename = 'model_best.pth'
                    if os.path.isfile(os.path.join(opt.model_dir, filename)):
                        shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'model_best.pth_bak'))                   
                    utils.save_checkpoint(state, opt.model_dir, filename=filename)

                model.train()
                margin.train()
                losses.reset()
                class_losses.reset()
                penalty_losses.reset()
                segment_class_losses.reset()
                   