import argparse
import os
import time

import torch
from dataset import PSGClsDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from trainer import BaseTrainer
from model_single import Group_vit, clip_tokenize
from sampler import MultilabelBalancedRandomSampler
import clip

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='test_small')
parser.add_argument('--epoch', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)

args = parser.parse_args()

savename = f'{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# loading model
# model = resnet50(pretrained=True)
# model.fc = torch.nn.Linear(2048, 56)
# model.cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
clip_model.requires_grad_(False)
text = clip_tokenize().detach().to(device)
print('CLIP for the text tokenization...', flush=True)
'''
model = Group_vit(
    image_size = 224,
    patch_size = 16,
    num_classes = 56,
    dim = 512,
    depth_T = 3,
    depth_G = 1,
    heads = 8,
    mlp_dim = 1024,
    vocab_size=100,
    dim_head=64,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

'''
model = Group_vit(
    image_size = 224, # for clip pretrain is useless
    patch_size = 16,  # for clip pretrain is useless
    num_classes = 56, 
    dim = 1024, # 512
    depth_T = 3,
    depth_G = 1,
    heads = 16, # 8
    mlp_dim = 4096, # 1024
    vocab_size=100,
    dim_head=64, # 64, 80 for huge
    dropout = 0.3, # 0.1
    emb_dropout = 0.3 # 0.1
).to(device)


print('Model parameters: ',sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Model Loaded...', flush=True)


# loading dataset
train_dataset = PSGClsDataset(stage='train', preprocess=preprocess)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                            #   shuffle=True,
                              sampler=MultilabelBalancedRandomSampler(train_dataset.get_all_label_for_sampler()),
                              num_workers=8,
                              pin_memory=True)

val_dataset = PSGClsDataset(stage='val', preprocess=preprocess)
val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)

test_dataset = PSGClsDataset(stage='test', preprocess=preprocess)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=8)
print('Data Loaded...', flush=True)


# loading trainer
trainer = BaseTrainer(clip_model,
                      model,
                      train_dataloader,
                      text,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch)
evaluator = Evaluator(clip_model, model, text, k=3)

# train!
print('Start Training...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0
for epoch in range(0, args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # show log
    print(
        '{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format(savename, (epoch + 1), int(time.time() - begin_epoch),
                train_metrics['train_loss'], val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
        flush=True)

    # save model
    if val_metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = val_metrics['mean_recall']
        
print(best_val_recall)
print('Training Completed...', flush=True)

'''
# saving result!
print('Loading Best Ckpt...', flush=True)
# checkpoint = torch.load(f'checkpoints/{savename}_best_28.50.ckpt')
checkpoint = torch.load("checkpoints/group_ViT-L-14@336_huge_1600_cls1_amsgrad_balanceDataCycle_e64_lr0.0005_bs32_m0.9_wd0.0005_best.ckpt")
model.load_state_dict(checkpoint)
test_evaluator = Evaluator(clip_model, model, text, k=3)
check_metrics = test_evaluator.eval_recall(val_dataloader)
best_val_recall = check_metrics['mean_recall']
if best_val_recall == check_metrics['mean_recall']:
    print('Successfully load best checkpoint with acc {:.2f}'.format(
        100 * best_val_recall),
          flush=True)
else:
    print('Fail to load best checkpoint')
result = test_evaluator.submit(test_dataloader)

# save into the file
with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)
'''