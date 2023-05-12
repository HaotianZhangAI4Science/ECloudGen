import argparse
import shutil
from utils.train import *
from utils.chem import *
import os.path as osp
from tqdm import tqdm
from models.ecloud2smi import ECloud2Mol
from utils.dataset import ECloudSMI, ECloudSMI_collate_fn
from torch.utils.data import Dataset, DataLoader
import logging
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn

logging.getLogger("moleculekit").setLevel(logging.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/train.yml')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--logdir', type=str, default='./logs')

args = parser.parse_args()

config = load_config(args.config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
seed_all(config.train.seed)
log_dir = get_new_log_dir(args.logdir, prefix=config_name)
logger = get_logger('train', log_dir)
logger.info(args)
logger.info(config)
shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
shutil.copytree('./models', os.path.join(log_dir, 'models'))
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

model = ECloud2Mol(vocab_dim=32)
optimizer = get_optimizer(config.train.optimizer, model)
scheduler = get_scheduler(config.train.scheduler, optimizer)

train = read_pkl('./data/zinc/0.pkl')
test = read_pkl('./data/test.pkl')
test_pairs = [i[1] for i in test]

test_set = ECloudSMI(mol_paths=test_pairs,data_base=config.train.data_base)
train_set = ECloudSMI(smi_list=train)
train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, collate_fn=ECloudSMI_collate_fn)
val_loader = DataLoader(test_set, batch_size=20, shuffle=True, num_workers=16, drop_last=False,collate_fn=ECloudSMI_collate_fn)

def evaluate(model, val_loader, logger, epoch, verbose=1):
    model.eval()
    eval_start = time.time()
    eval_losses = []
    for batch in val_loader:
        targets = pack_padded_sequence(batch[1],  batch[2], batch_first=True)[0]
        decipher = model(batch[0], batch[1], batch[2])
        loss = decipher_loss(decipher,targets)
        eval_losses.append(loss.item())    
    average_loss = np.mean(eval_losses)
    return average_loss

def load(checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
    
    epoch = str(epoch) if epoch is not None else ''
    checkpoint = os.path.join(checkpoint,epoch)
    logger.info("Load checkpoint from %s" % checkpoint)

    state = torch.load(checkpoint, map_location=args.device)   
    model.load_state_dict(state["model"])
    #self._model.load_state_dict(state["model"], strict=False)
    #best_loss = state['best_loss']
    #start_epoch = state['cur_epoch'] + 1

    if load_scheduler:
        scheduler.load_state_dict(state["scheduler"])
        
    if load_optimizer:
        optimizer.load_state_dict(state["optimizer"])
        if args.device == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(args.device)
    return state['loss']

def save(model, optimizer, scheduler, epoch, loss, check_point):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'loss': np.mean(epoch_loss)
        },check_point)

best_loss = 10
val_losses = [best_loss]
logger.info('start training...')
decipher_loss = nn.CrossEntropyLoss()
for epoch in range(10):
    epoch_loss = []
    batch_cnt = 0

    if config.resume_train.resume_train:
        ckpt_name = config.resume_train.ckpt_name
        start_epoch = int(config.resume_train.start_epoch)
        best_loss = load(osp.join(config.resume_train.checkpoint_path,ckpt_name))
        
    for batch in train_loader:
        batch_cnt += 1
        targets = pack_padded_sequence(batch[1],  batch[2], batch_first=True)[0]
        decipher = model(batch[0], batch[1], batch[2])
        loss = decipher_loss(decipher,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if (epoch==0 and batch_cnt <= 10):
            logger.info('Training Epoch %d | Loss %.6f'%(epoch, loss.item()))
        if batch_cnt % 10000 == 0:
            show_loss = np.mean(epoch_loss)
            ckpt_path = os.path.join(ckpt_dir, 'train_%d_btc_%d.pt' % (int(epoch),batch_cnt))
            logger.info('Training Epoch %d | Batch Count %d | Loss %.6f'%(epoch, batch_cnt, show_loss))
            save(model, optimizer, scheduler, epoch, best_loss, ckpt_path)
    
    if epoch == 0 or np.mean(epoch_loss)<best_loss:
        best_loss = np.mean(epoch_loss)
        ckpt_path = os.path.join(ckpt_dir, 'train_%d.pt' % int(epoch))
        save(model, optimizer, scheduler, epoch, best_loss, ckpt_path)
        
    logger.info('Training Epoch %d | Loss %.6f'%(epoch, np.mean(epoch_loss)))
    average_eval_loss = evaluate(model, val_loader, logger, epoch, verbose=1)
    logger.info('Evaluation Epoch %d | Loss %.6f'%(epoch, average_eval_loss))
    val_losses.append(average_eval_loss)

    if config.train.scheduler.type=="plateau":
        scheduler.step(average_eval_loss)
    else:
        scheduler.step()
    if val_losses[-1] < val_losses[-2]:
        ckpt_path = os.path.join(ckpt_dir, 'val_%d.pt' % int(epoch))
        save(model, optimizer, scheduler, epoch, best_loss, ckpt_path)

