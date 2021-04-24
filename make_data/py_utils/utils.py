import numpy as np
import time
import torch
import os
import sys
import logging
import time
from datetime import timedelta
from collections import OrderedDict
import math

def choose_quantization_params(w_abs_max, qmax):
    w_abs_max = float(qmax) / w_abs_max 
    Q = torch.floor(torch.log2(w_abs_max))    
    return Q

def quantize(qparams, fdata, qBit, device):
    bound = math.pow(2.0, qBit-1)
    min_val = - bound
    max_val = bound - 1
    if len(fdata.size()) == 1:
        fdata = fdata.unsqueeze(1)    
    times = torch.pow(torch.FloatTensor([2.0]).to(device), qparams)
    transformed_val = fdata * times
    clipped_value = torch.round(torch.clamp(transformed_val, min_val, max_val))
    if qBit == 8:
        return clipped_value.to(torch.int8)
    elif qBit == 16:
        return clipped_value.to(torch.int16)
    else:
        return clipped_value

def dequantize(qparams, fixed_data, device):    
    times = torch.pow(torch.FloatTensor([2.0]).to(device), -qparams)
    fixed_data = fixed_data.to(torch.float32)
    transformed_val = fixed_data * times
    return transformed_val

def quantize_model(model, device):
    qmax = 127
    params = None
    state_dict_dequant = OrderedDict()
    for k, v in model.state_dict().items(): 
        if "weight" in k:
            w_abs_max = torch.max(torch.abs(v), 1, keepdim=True)[0]
            params = choose_quantization_params(w_abs_max, qmax)
            fixed_w = quantize(params, v, 8, device)
            dequant_w = dequantize(params, fixed_w, device)
            state_dict_dequant[k] = dequant_w
        elif "bias" in k:
            fixed_b = quantize(params, v, 16, device)            
            dequant_b = dequantize(params, fixed_b, device)
            dequant_b = dequant_b.squeeze()
            state_dict_dequant[k] = dequant_b
    model.load_state_dict(state_dict_dequant)
    return model

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def convfft(a, b):
    N = len(a)
    M = len(b)
    YN = N + M - 1
    FFT_N = 2 ** (int(np.log2(YN)) + 1)
    afft = np.fft.fft(a, FFT_N)
    bfft = np.fft.fft(b, FFT_N)
    abfft = afft * bfft
    y = np.fft.ifft(abfft).real[:YN]
    return y

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)
        
def create_output_dir(opt):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    filepath = os.path.join(expr_dir, 'main.log')

    # Safety check
    if os.path.exists(filepath) and opt.resume == "":
        logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # quite down visdom
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger

def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm_(params, clip_th)
    return (not np.isfinite(befgad) or (befgad > ignore_th))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_np(x):
    return x.data.cpu().numpy()

def accuracy(output, target):
    pred = output.data.max(1)[1] # get the index of the max log-probability
    batch_size = target.size(0)
    correct = pred.eq(target.data).cpu().sum()
    correct *= (100.0 / batch_size)
    return correct


def compute_acc(output, target):
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    batch_size = target.size(0)
    correct *= (100.0 / batch_size)
    return correct         

def save_checkpoint(state, save_path, is_best=False, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename),
                        os.path.join(save_path, 'model_best.pth.tar'))

def adjust_learning_rate_by_factor(optimizer, lr, factor):
    """Adjusts the learning rate according to the given factor"""
    lr = lr * factor
    lr = max(lr, 0.000005)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
                                        
def test_edit_distance():
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=False)

    with tf.Session(graph=graph) as session:
        truthTest = sparse_tensor_feed([[0,1,2], [0,1,2,3,4]])
        hypTest = sparse_tensor_feed([[3,4,5], [0,1,2,2]])
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run([editDist], feed_dict=feedDict)
        print(dist)

def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
    in the list being a list or array with the values of the target sequence
    (e.g., the integer values of a character map for an ASR target string)
    See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
    for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))
        


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
	    sequence.
		   If maxlen is provided, any sequence longer than maxlen is truncated to
		  maxlen. Truncation happens off either the beginning or the end
		  (default) of the sequence. Supports post-padding (default) and
		  pre-padding.

		  Args:
			  sequences: list of lists where each element is a sequence
			  maxlen: int, maximum length
			  dtype: type to cast the resulting sequence.
			  padding: 'pre' or 'post', pad either before or after each sequence.
			  truncating: 'pre' or 'post', remove values from sequences larger
			  than maxlen either in the beginning or in the end of the sequence
			  value: float, value to pad the sequences to the desired value.
		  Returns
			  x: numpy array with dimensions (number_of_sequences, maxlen)
			  lengths: numpy array with the original sequence lengths
	  '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

	  # take the sample shape from the first non empty sequence
	  # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
        break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

		# check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
        raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
						 (trunc.shape[1:], idx, sample_shape))

    if padding == 'post':
        x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
        x[idx, -len(trunc):] = trunc
    else:
        raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape
            