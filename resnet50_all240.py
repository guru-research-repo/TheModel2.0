import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import yaml
import json
from utils import *
from pipeline import build_transform_pipeline
import matplotlib.pyplot as plt
import argparse


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.resnet_model = torchvision.models.resnet50(pretrained=False)
        self.model = nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        cer = nn.CrossEntropyLoss()
        loss = cer(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if epoch == 0 and batch_idx == 0:
            print("Sample labels from first batch:", target[:10].tolist())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

    losses = np.array(losses)
    average_loss = np.mean(losses)
    return average_loss

def test(model, device, loader):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = (correct*1.0) / len(loader.dataset)
    return loss / len(loader), accuracy
######################################################################################################
# import matplotlib.pyplot as plt
# def visualize_batch(loader, identity_map):
#     rev_map = {v: k for k, v in identity_map.items()}
#     data_iter = iter(loader)
#     images, labels = next(data_iter)
#     for i in range(5):
#         img = images[i].permute(1, 2, 0).cpu().numpy()
#         label = labels[i].item()
#         plt.imshow(img)
#         plt.title(f"Label {label}: {rev_map[label]}")
#         plt.axis('off')
#         plt.savefig("justchecking.png")
######################################################################################################


def main():
    phases = [4, 8, 16, 32, 64, 128]
    total_epochs_per_phase = 40

    parser = argparse.ArgumentParser('description=PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=48, metavar='N')
    parser.add_argument('--test_batch_size', type=int, default=48, metavar='N')
    parser.add_argument('--epochs', type=int, default=40, metavar='N')
    parser.add_argument('--initial_lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N')
    parser.add_argument('--workers', type=int, default=4, metavar='N')

    parser.add_argument('--dataset_name', type=str, default='faces', help='faces or celeb')
    parser.add_argument('--identity_count', type=int, default=4)
    #parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    save_dir = os.path.join(args.out_path, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    transform = build_transform_pipeline(config)

    train_accs, val_upright_accs, val_inverted_accs = [], [], []
    model = None
    best_val_acc = 0.0

    for phase_idx, identity_count in enumerate(phases):
        print(f"\n=== Phase {phase_idx+1}: Training with {identity_count} identities ===")
        classes = identity_count

        train_set = load_dataset(args.dataset_name, identity=classes, task='train', transform=build_transform_pipeline(config, inversion=None))
        val_upright = load_dataset(args.dataset_name, identity=classes, task='valid', transform=build_transform_pipeline(config, inversion=False))
        val_inverted = load_dataset(args.dataset_name, identity=classes, task='valid', transform=build_transform_pipeline(config, inversion=True))
        test_upright = load_dataset(args.dataset_name, identity=classes, task='test', transform=build_transform_pipeline(config, inversion=False))
        test_inverted = load_dataset(args.dataset_name, identity=classes, task='test', transform=build_transform_pipeline(config, inversion=True))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        ###################################################################################################
        # identity_map = {'EmmanuelMacron': 0, 'GalGadot': 1, 'JohnLegend': 2, 'OprahWinfrey': 3, 'AdamRippon': 4, 'Lin-ManuelMiranda': 5, 'SherylSandberg': 6, 'SimoneBiles': 7, 'AliciaKeys': 8, 'ConstanceWu': 9, 'DonaldGlover': 10, 'EdSheeran': 11, 'GuillermoDelToro': 12, 'LiuWen': 13, 'SadiqKhan': 14, 'TildaSwinton': 15}
        # visualize_batch(train_loader, identity_map)
        ###################################################################################################
        val_loader_upright = torch.utils.data.DataLoader(val_upright, batch_size=args.batch_size, shuffle=True)
        val_loader_inverted = torch.utils.data.DataLoader(val_inverted, batch_size=args.batch_size, shuffle=True)
        test_loader_upright = torch.utils.data.DataLoader(test_upright, batch_size=args.batch_size, shuffle=True)
        test_loader_inverted = torch.utils.data.DataLoader(test_inverted, batch_size=args.batch_size, shuffle=True)

        if model is None:
            model = Model(classes).to(device)
        else:
            model.fc2 = nn.Linear(1000, classes).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-3)

        low_val_upright_loss = np.inf
        loss_file = open(os.path.join(save_dir, 'loss.txt'), 'w')

        for epoch in range(total_epochs_per_phase):
            train_loss = train(args, model, device, train_loader, optimizer, epoch)
            print("\nEvaluating on training set...")
            tr_loss, tr_acc = test(model, device, train_loader)
            print("\nEvaluating on upright validation set...")
            val_loss_u, val_acc_u = test(model, device, val_loader_upright)
            print("\nEvaluating on inverted validation set...")
            val_loss_i, val_acc_i = test(model, device, val_loader_inverted)

            print(f"Epoch {epoch+1}/{total_epochs_per_phase} | Train Acc: {tr_acc:.4f} | Val U: {val_acc_u:.4f} | Val I: {val_acc_i:.4f}")

            train_accs.append(tr_acc)
            val_upright_accs.append(val_acc_u)
            val_inverted_accs.append(val_acc_i)

            # Save metrics
            metrics = {
                'epoch': epoch+1,
                'phase': identity_count,
                'train_acc': tr_acc,
                'train_loss':tr_loss,
                'val_upright_acc': val_acc_u,
                'val_upright_loss':val_loss_u,
                'val_inverted_acc': val_acc_i,
                'val_inverted_loss':val_loss_i
            }
            with open(os.path.join(save_dir, f'metrics_phase{identity_count}_epoch{epoch}.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            # Save best model
            if val_loss_u < low_val_upright_loss:
                print("Saving new best model...")
                low_val_upright_loss = val_loss_u
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_dir, 'best_model.pth'))
                #torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

                # Also save a phase-specific best model
                phase_best_model_path = os.path.join(save_dir, f'best_model_phase{identity_count}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, phase_best_model_path)
                print(f"Saved phase-specific best model to {phase_best_model_path}")

            # Save last model checkpoint every epoch
            print("Save last model checkpoint every epoch")
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_phase{identity_count}_epoch{epoch+1}.pth'))

            loss_file.write(f"epoch{epoch}:{tr_loss},{tr_acc},{val_loss_u},{val_acc_u},{val_loss_i},{val_acc_i}\n")

    loss_file.close()

    ### Final Testing
    print("\nLoading best model...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}...")

    print("\nEvaluating on upright test set...")
    test_loss_u, test_acc_u = test(model, device, test_loader_upright)

    print("\nEvaluating on inverted test set...")
    test_loss_i, test_acc_i = test(model, device, test_loader_inverted)

    print(f'\nUpright Test Set Loss: {test_loss_u}')
    print(f'Upright Test Set Accuracy: {test_acc_u}')
    print(f'\nInverted Test Set Loss: {test_loss_i}')
    print(f'Inverted Test Set Accuracy: {test_acc_i}')

    # # Plot accuracy
    # epochs = list(range(1, total_epochs_per_phase * len(phases) + 1))
    # plt.figure(figsize=(12, 5))
    # plt.plot(epochs, train_accs, label='Train Acc')
    # plt.plot(epochs, val_upright_accs, label='Valid Upright Acc')
    # plt.plot(epochs, val_inverted_accs, label='Valid Inverted Acc')
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy vs Epochs across Curriculum Learning Phases")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(save_dir, "curriculum_accuracy.png"))
    # print("Saved accuracy plot!")

if __name__ == '__main__':
    main()
