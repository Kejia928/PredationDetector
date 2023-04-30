import torch.nn as nn
import torch as torch
from matplotlib import pyplot as plt
import time
import copy
import tqdm
import os
from SequenceDataset import SequenceDataset
from torch.utils.data import DataLoader

def train_model(model, criterion, batch_size, optimizer, num_epochs=25, is_inception=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_dataset = SequenceDataset(path='../video2img', json_path='../feeding_dataset/videoAnnotation_train.jsonl', drop=True, random=True)
        test_dataset = SequenceDataset(path='../video2img', json_path='../feeding_dataset/videoAnnotation_test.jsonl', drop=True, random=False)
        train_dataset.getAnnotation()
        test_dataset.getAnnotation()
        train_dataset.initDataset()
        test_dataset.initDataset()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        dataloaders = {
            'train': train_loader,
            'val': test_loader
        }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc="Run on dataset: "):
                inputs = inputs.unsqueeze(2)
                inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    path = 'runs'
    if not os.path.exists(path):
        os.mkdir(path)
    all_exp = os.listdir(path)
    all_exp.sort()
    # num = 0
    if all_exp is None:
        path = path + '/exp' + str(0)
        os.mkdir(path)
    else:
        # print(all_exp)
        # num = int(re.findall(r'\d+', all_exp[-1])[-1])
        # print(num)
        path = path + '/exp' + str(len(all_exp))
        os.mkdir(path)
    print("The result save in ", path)
    torch.save(best_model_wts, path + '/best.pt')

    train_acc_history_cpu = [t.cpu().numpy() for t in train_acc_history]
    val_acc_history_cpu = [t.cpu().numpy() for t in val_acc_history]

    # plot diagram
    plt.plot(range(num_epochs), train_acc_history_cpu, "g*-", label='train_acc')
    plt.plot(range(num_epochs), val_acc_history_cpu, "b*-", label='val_acc')
    plt.xlabel('num_epoch')
    plt.title('Train and Val acc')
    plt.legend()
    plt.savefig(path + '/acc.png')
    plt.clf()

    plt.plot(range(num_epochs), train_loss_history, "k*-", label='train_loss')
    plt.plot(range(num_epochs), val_loss_history, "y*-", label='test_loss')
    plt.xlabel('num_epoch')
    plt.title('Train and Val loss')
    plt.legend()
    plt.savefig(path + '/loss.png')

    return model, val_acc_history

