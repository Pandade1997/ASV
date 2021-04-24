import os
import time
import shutil
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel

from data.test_data_loader import DeepSpeakerTestDataset, DeepSpeakerTestDataLoader
from data.ge2e_data_loader import DeepSpeakerDataset, DeepSpeakerDataLoader
from model.model import model_select, load
from model.margin import margin_select
from train.base_trainer import reduce_loss_dict, union_evaluate
from utils import utils


def train(opt, logging):

    ## Data Prepare ##
    if opt.main_proc:
        logging.info("Building dataset")
                                           
    train_dataset = DeepSpeakerDataset(opt, os.path.join(opt.dataroot, 'dev'))
    train_loader = DeepSpeakerDataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
             
    val_dataset = DeepSpeakerTestDataset(opt, os.path.join(opt.dataroot, 'test'))
    val_loader = DeepSpeakerTestDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False, pin_memory=True)
    
    opt.in_size = train_dataset.in_size
    opt.out_size = train_dataset.class_nums  
    print('opt.in_size {} opt.out_size {}'.format(opt.in_size, opt.out_size))  
                                           
    if opt.main_proc:
        logging.info("Building dataset Sucessed")
    
    ##  Building Model ##
    if opt.main_proc:
        logging.info("Building Model")
    
    opt.model_type = opt.model_type_1
    model_1 = model_select(opt, seq_training=False) ## rnn ge2e
    opt.model_type = opt.model_type_2
    model_2 = model_select(opt, seq_training=False) ## cnn class
    embedding_size = opt.embedding_size
    opt.embedding_size = 2 * embedding_size
    margin = margin_select(opt)
    opt.embedding_size = embedding_size
    
    if opt.resume_1:
        model_1, opt.total_iters = load(model_1, opt.resume_1, 'state_dict')    
    if opt.resume_2:
        model_2, opt.total_iters = load(model_2, opt.resume_2, 'state_dict')
        margin, opt.total_iters = load(margin, opt.resume_2, 'margin_state_dict')
        
    if opt.resume:
        model_1, opt.total_iters = load(model_1, opt.resume, 'state_dict_1')
        model_2, opt.total_iters = load(model_2, opt.resume, 'state_dict_2')
        margin, opt.total_iters = load(margin, opt.resume, 'margin_state_dict')
        
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    if opt.optim_type == 'sgd':
        optimizer = optim.SGD([
            {'params': model_1.parameters(), 'weight_decay': 5e-4},
            {'params': model_2.parameters(), 'weight_decay': 5e-4},
            {'params': margin.parameters(), 'weight_decay': 5e-4},
        ], lr=opt.lr, momentum=0.9, nesterov=True)
    elif opt.optim_type == 'adam':
        optimizer = optim.Adam([
            {'params': model_1.parameters(), 'weight_decay': 5e-4},
            {'params': model_2.parameters(), 'weight_decay': 5e-4},
            {'params': margin.parameters(), 'weight_decay': 5e-4},
        ], lr=opt.lr, betas=(opt.beta1, 0.999))
        
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=opt.lr_reduce_step, gamma=opt.lr_reduce_factor, last_epoch=-1)
        
    model_1.to(opt.device)
    model_2.to(opt.device)
    margin.to(opt.device)
    
    if opt.distributed:
        model_1 = DistributedDataParallel(model_1, device_ids=[opt.local_rank], output_device=opt.local_rank)
        model_2 = DistributedDataParallel(model_2, device_ids=[opt.local_rank], output_device=opt.local_rank)
        margin  = DistributedDataParallel(margin, device_ids=[opt.local_rank], output_device=opt.local_rank)
    if opt.main_proc:
        print(model_1)
        print(model_2)
        print(margin)
        logging.info("Building Model Sucessed") 
        
    best_perform_acc = 1.0
    
    losses = utils.AverageMeter()
    class_losses = utils.AverageMeter()
    embedding_losses = utils.AverageMeter()
    penalty_losses = utils.AverageMeter()

    # Initial performance
    if opt.main_proc:
        EER = union_evaluate(opt, model_1, model_2, val_loader, logging)
        best_perform_acc = EER
        print('>>Start performance: EER = {}<<'.format(best_perform_acc))
    
    save_model = model_1
    if isinstance(model_1, DistributedDataParallel):
        save_model = model_1.module
                            
    # Start Training
    total_iters = opt.total_iters
    for epoch in range(1, opt.total_epoch + 1):
        while True:
            model_1.train()
            model_2.train()
            margin.train()
            for i, (data) in enumerate(train_loader, start=0):
                if i == len(train_loader):
                    break

                optimizer.zero_grad()

                # Perform forward and Obtain the loss
                feature_input, spk_ids = data               
                feature_input = feature_input.to(opt.device)
                label = spk_ids.to(opt.device).squeeze(0)
                
                output_1, attn_1, w_1, b_1 = model_1(feature_input)                                
                output_2, attn_2, w_2, b_2 = model_2(feature_input)                
                margin_input = torch.cat((output_1, output_2), dim=1)
                margin_output = margin(margin_input, label)
                
                output_1 = save_model.normalize(output_1)  
                sim_matrix_out = save_model.similarity(output_1, w_1, b_1)  
                embedding_loss = opt.embedding_loss_lamda / (opt.speaker_num * opt.utter_num) * save_model.loss_cal(sim_matrix_out) 
                if opt.att_type == 'multi_attention' and attn_1 is not None:
                    penalty_loss = opt.penalty_loss_lamda * save_model.penalty_loss_cal(attn_1)
                else:
                    penalty_loss = 0
                class_loss = opt.class_loss_lamda * criterion(margin_output, label)
                loss = embedding_loss + penalty_loss + class_loss
                
                loss_dict_reduced = reduce_loss_dict(opt, {'embedding_loss': embedding_loss, 'penalty_loss': penalty_loss, 'class_loss': class_loss})                
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
                embedding_loss_value = loss_dict_reduced['embedding_loss'].item()
                penalty_loss_value = loss_dict_reduced['penalty_loss'].item()
                class_loss_value = loss_dict_reduced['class_loss'].item()

                # Check the loss and avoid the invaided loss
                inf = float("inf")
                if loss_value == inf or loss_value == -inf:
                    print("WARNING: received an inf loss, setting loss value to 0")
                    loss_value = 0
                    embedding_loss_value = 0
                    penalty_loss_value = 0
                    class_loss_value = 0
                    continue

                # Perform backward and Check and update the grad
                loss.backward()
                if utils.check_grad(model_1.parameters(), opt.clip_grad, opt.ignore_grad) or utils.check_grad(model_2.parameters(), opt.clip_grad, opt.ignore_grad):
                    if opt.main_proc:
                        logging.info('Not a finite gradient or too big, ignoring')
                    optimizer.zero_grad()
                    continue
                optimizer.step()
    
                total_iters += opt.num_gpus

                # Update the loss for logging
                losses.update(loss_value)
                embedding_losses.update(embedding_loss_value)
                penalty_losses.update(penalty_loss_value)
                class_losses.update(class_loss_value)

                # Print the performance on the training dateset 'opt': opt, 'learning_rate': lr,
                if total_iters % opt.print_freq == 0:
                    scheduler.step(total_iters)
                    if opt.main_proc:
                        lr = scheduler.get_lr()
                        if isinstance(lr, list):
                            lr = max(lr)
                        logging.info('==> Train set steps {} lr: {:.6f}, loss: {:.4f} [ class: {:.4f}, embedding: {:.4f}, penalty_loss {:.4f}]'.format(
                                     total_iters, lr, losses.avg, class_losses.avg, embedding_losses.avg, penalty_losses.avg))
        
                        if opt.distributed:
                            model_state_dict_1 = model_1.module.state_dict()
                            model_state_dict_2 = model_2.module.state_dict()
                            margin_state_dict = margin.module.state_dict()
                        else:
                            model_state_dict_1 = model_1.state_dict()
                            model_state_dict_2 = model_2.state_dict()
                            margin_state_dict = margin.state_dict()
                        state = {'state_dict_1': model_state_dict_1, 'total_iters': total_iters,
                                 'state_dict_2': model_state_dict_2, 'margin_state_dict': margin_state_dict}
                        filename = 'newest_model.pth'
                        if os.path.isfile(os.path.join(opt.model_dir, filename)):
                            shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'newest_model.pth_bak'))
                        utils.save_checkpoint(state, opt.model_dir, filename=filename)

                # Validate the trained model
                if total_iters % opt.validate_freq == 0:
                    EER = union_evaluate(opt, model_1, model_2, val_loader, logging)
                    ##scheduler.step(EER)
                    
                    if opt.main_proc and EER < best_perform_acc:
                        best_perform_acc = EER
                        print("Found better validated model (EER = %.3f), saving to model_best.pth" % (best_perform_acc))
                        
                        if opt.distributed:
                            model_state_dict_1 = model_1.module.state_dict()
                            model_state_dict_2 = model_2.module.state_dict()
                            margin_state_dict = margin.module.state_dict()
                        else:
                            model_state_dict_1 = model_1.state_dict()
                            model_state_dict_2 = model_2.state_dict()
                            margin_state_dict = margin.state_dict()
                        state = {'state_dict_1': model_state_dict_1, 'total_iters': total_iters,
                                 'state_dict_2': model_state_dict_2, 'margin_state_dict': margin_state_dict}
                        
                        filename = 'model_best.pth'
                        if os.path.isfile(os.path.join(opt.model_dir, filename)):
                            shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'model_best.pth_bak'))
                        utils.save_checkpoint(state, opt.model_dir, filename=filename)                             
    
                    model_1.train()
                    model_2.train()
                    margin.train()
                    losses.reset()
                    class_losses.reset()
                    embedding_losses.reset()
                    penalty_losses.reset()
    
                if total_iters > opt.max_iters and opt.main_proc:
                    logging.info('finish training, steps is  {}'.format(total_iters))
                    return model_1
