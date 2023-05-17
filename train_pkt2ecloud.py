import argparse
import shutil
from utils.train import *
from utils.chem import *
import os.path as osp
from tqdm import tqdm
from models.pkt2ecloud import Pkt2ECloud
from utils.dataset import ECloud
from torch.utils.data import Dataset, DataLoader
import logging


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

pkt2ecloud = Pkt2ECloud(device=args.device).to(args.device)
optimizer = get_optimizer(config.train.optimizer, pkt2ecloud)
scheduler = get_scheduler(config.train.scheduler, optimizer)

logger.info('Start to load data...')

train = read_pkl('./data/train.pkl')
test = read_pkl('./data/test.pkl')

logger.info('The loaded protein-ligand pairs used for training are {}'.format(len(train)))

train_set, test_set = ECloud(train, config.dataset.data_base), ECloud(test, config.dataset.data_base)
train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)
val_loader = DataLoader(test_set, batch_size=20, shuffle=True, num_workers=16, drop_last=False)

def evaluate(model, val_loader, logger, epoch, verbose=1):
    model.eval()
    eval_start = time.time()
    eval_losses = []
    for batch in val_loader:
        recon, mu, logvar = model(batch[0].to(args.device))
        loss = loss_function(recon, batch[1].to(args.device), mu, logvar)
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
            'model': pkt2ecloud.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'loss': np.mean(epoch_loss)
        },check_point)


best_loss = 10
val_losses = [best_loss]
logger.info('start training...')
for epoch in range(300):
    epoch_loss = []
    batch_cnt = 0

    if config.resume_train.resume_train:
        ckpt_name = config.resume_train.ckpt_name
        start_epoch = int(config.resume_train.start_epoch)
        best_loss = load(osp.join(config.resume_train.checkpoint_path,ckpt_name))
        
        

    for batch in train_loader:
        batch_cnt += 1
        recon, mu, logvar = pkt2ecloud(batch[0].to(args.device))
        loss = loss_function(recon, batch[1].to(args.device), mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if (epoch==0 and batch_cnt <= 10):
            logger.info('Training Epoch %d | Loss %.6f'%(epoch, loss.item()))
    
    if epoch == 0 or np.mean(epoch_loss)<best_loss:
        best_loss = np.mean(epoch_loss)
        ckpt_path = os.path.join(ckpt_dir, 'train_%d.pt' % int(epoch))
        save(pkt2ecloud, optimizer, scheduler, epoch, best_loss, ckpt_path)
        
    logger.info('Training Epoch %d | Loss %.6f'%(epoch, np.mean(epoch_loss)))
    average_eval_loss = evaluate(pkt2ecloud, val_loader, logger, epoch, verbose=1)
    logger.info('Evaluation Epoch %d | Loss %.6f'%(epoch, average_eval_loss))
    val_losses.append(average_eval_loss)

    if config.train.scheduler.type=="plateau":
        scheduler.step(average_eval_loss)
    else:
        scheduler.step()
    if val_losses[-1] < val_losses[-2]:
        ckpt_path = os.path.join(ckpt_dir, 'val_%d.pt' % int(epoch))
        save(pkt2ecloud, optimizer, scheduler, epoch, best_loss, ckpt_path)
