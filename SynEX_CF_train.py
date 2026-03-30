import warnings
warnings.filterwarnings("ignore")
import os
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils import *
from SynEx_CF import SynEx_CF
from predict import pk_load

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--gpus', type=str, default='0,1,2,3')
parser.add_argument('--fold', type=int, default=12)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--pre_trained', type=str, default=False)
parser.add_argument('--data', type=str, default='her2st')
parser.add_argument('--seed', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=401)
parser.add_argument('--name', type=str, default='hist2ST')
parser.add_argument('--prune', type=str, default='Grid')
parser.add_argument('--neighbor', type=int, default=8)
parser.add_argument('--train_samples', type=str, default="A1,B1,C1,D1,E1,F1,G2,H1")
parser.add_argument('--freeze_vit', action='store_true')
parser.add_argument('--freeze_vit_layers', type=int, default=11)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=f'./logs/NewHGGEP_{args.data}_{args.name}')
        print(f"Starting training with world_size: {world_size}, rank: {rank}")

    if args.train_samples:
        train_samples = args.train_samples.split(',')
    else:
        train_samples = None

    trainset = pk_load(args.fold, 'train', False, args.data, neighs=args.neighbor, prune=args.prune, train_samples=train_samples)
    valset = pk_load(args.fold, 'train', False, args.data, neighs=args.neighbor, prune=args.prune, train_samples=train_samples)
    testset = pk_load(args.fold, 'test', False, args.data, neighs=args.neighbor, prune=args.prune, train_samples=train_samples)

    label = None
    if args.fold in [5, 11, 17, 23, 26, 30] and args.data == 'her2st':
        label = testset.label[testset.names[0]]

    n_genes = 785
    if args.data == 'cscc':
        args.name += '_cscc'
        n_genes = 171
    elif args.data == 'BDS':
        args.name += '_BDS'
        n_genes = 100

    model = SynEx_CF(
        n_genes=n_genes,
        learning_rate=args.lr,
        label=label,
        freeze_vit=args.freeze_vit,
        freeze_vit_layers=args.freeze_vit_layers
    )

    model = model.to(device)

    if args.pre_trained:
        model.load_state_dict(torch.load(args.pre_trained, map_location=device))

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    train_sampler = DistributedSampler(trainset, shuffle=True)
    val_sampler = DistributedSampler(valset, shuffle=False)
    test_sampler = DistributedSampler(testset, shuffle=False)

    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=1, num_workers=0, sampler=val_sampler)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, sampler=test_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    mse_loss = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_pcc = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            patches, positions, exp, adj, *_, centers = batch
            patches = patches.to(device)
            positions = positions.to(device)
            exp = exp.to(device)
            adj = adj.to(device)

            labels = None
            if len(batch) > 6:
                labels = batch[6].to(device)
            else:
                labels = torch.full((1, patches.shape[1]), -1, dtype=torch.long, device=device)

            pred = model(patches, positions, labels, adj)

            loss = mse_loss(pred.squeeze(0), exp.squeeze(0))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_np = pred.squeeze(0).cpu().detach().numpy()
            exp_np = exp.squeeze(0).cpu().numpy()
            pcc = model.module.calculate_pcc(pred_np, exp_np) if hasattr(model, 'module') else model.calculate_pcc(pred_np, exp_np)
            train_pcc += pcc

        scheduler.step()

        train_loss /= len(train_loader)
        train_pcc /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_pcc = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                patches, positions, exp, adj, *_, centers = batch
                patches = patches.to(device)
                positions = positions.to(device)
                exp = exp.to(device)
                adj = adj.to(device)

                labels = None
                if len(batch) > 6:
                    labels = batch[6].to(device)
                else:
                    labels = torch.full((1, patches.shape[1]), -1, dtype=torch.long, device=device)

                pred = model(patches, positions, labels, adj)

                loss = mse_loss(pred.squeeze(0), exp.squeeze(0))
                val_loss += loss.item()

                pred_np = pred.squeeze(0).cpu().detach().numpy()
                exp_np = exp.squeeze(0).cpu().numpy()
                pcc = model.module.calculate_pcc(pred_np, exp_np) if hasattr(model, 'module') else model.calculate_pcc(pred_np, exp_np)
                val_pcc += pcc

        val_loss /= len(val_loader)
        val_pcc /= len(val_loader)

        if rank == 0:
            if writer:
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('train_pcc', train_pcc, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                writer.add_scalar('val_pcc', val_pcc, epoch)

            print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.6f}, Train PCC: {train_pcc:.6f}, Val Loss: {val_loss:.6f}, Val PCC: {val_pcc:.6f}")

            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                os.makedirs('./model/NewHGGEP/', exist_ok=True)
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                          f'./model/NewHGGEP/{args.name}_fold{args.fold}-{epoch+1:02d}-{val_loss:.2f}.pth')

    model.eval()
    test_loss = 0.0
    test_pcc = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            patches, positions, exp, adj, *_, centers = batch
            patches = patches.to(device)
            positions = positions.to(device)
            exp = exp.to(device)
            adj = adj.to(device)

            labels = None
            if len(batch) > 6:
                labels = batch[6].to(device)
            else:
                labels = torch.full((1, patches.shape[1]), -1, dtype=torch.long, device=device)

            pred = model(patches, positions, labels, adj)

            loss = mse_loss(pred.squeeze(0), exp.squeeze(0))
            test_loss += loss.item()

            pred_np = pred.squeeze(0).cpu().detach().numpy()
            exp_np = exp.squeeze(0).cpu().numpy()
            pcc = model.module.calculate_pcc(pred_np, exp_np) if hasattr(model, 'module') else model.calculate_pcc(pred_np, exp_np)
            test_pcc += pcc

    test_loss /= len(test_loader)
    test_pcc /= len(test_loader)

    if rank == 0:
        print(f"Test Loss: {test_loss:.6f}, Test PCC: {test_pcc:.6f}")

    if writer:
        writer.close()

    cleanup()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    gpus = args.gpus.split(',')
    world_size = len(gpus)

    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
