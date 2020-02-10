import torch
import numpy as np
import tqdm
from collections import defaultdict
from itertools import chain
from sklearn.metrics import accuracy_score, f1_score

BEST_SCORE = 0.0


def train_epoch(net, train_loader, optimizer, criterion, ep, scheduler=None, flatten=False, MODE='max', rnn=False):
    net.train()

    running_loss = 0.0
    running_acc = 0.0

    y_pred_ = []
    y_true_ = []

    for img, y_true, image_id in tqdm.tqdm(train_loader, desc='train'):
        if MODE == 'min':
            # slice based-on min_depth
            img = img[:, :, :train_loader.dataset.min_depth]

        img = img.cuda()
        y_true = y_true.cuda()

        if flatten:
            bs, c, depth, h, w = img.shape
            img = img.view(-1, c*depth, h, w)
        if rnn:
            img = img.permute(0, 2, 1, 3, 4)

            state = None
            loss = 0.0
            for t in range(img.size(1)):
                outputs, state = net(img[:, t], state)
                loss += criterion(outputs, torch.max(y_true, 1)[1])

        else:
            outputs = net(img)
            loss = criterion(outputs, torch.max(y_true, 1)[1])

        _, y_pred = torch.max(outputs, 1)
        _, y_true = torch.max(y_true, 1)

        y_pred_.append(y_pred.detach().cpu().numpy())
        y_true_.append(y_true.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())

        running_loss += loss.item()
        running_acc += acc.item()

    running_loss /= len(train_loader)
    running_acc /= len(train_loader)

    # f1-score
    f1 = f1_score(list(chain(*y_true_)), list(chain(*y_pred_)))

    print(f'[Train] ep : {ep} loss : {running_loss:.4f} \
          acc : {running_acc:.4f} f1 : {f1:.4f}')

    if scheduler is not None:
        scheduler.step()


def evaluate(net, test_loader, criterion, ep, logging=False, flatten=False, MODE='max', rnn=False):
    net.eval()

    res = defaultdict(list)

    running_loss = 0.0
    running_acc = 0.0

    y_pred_ = []
    y_true_ = []

    for img, y_true, image_id in tqdm.tqdm(test_loader, desc='test'):
        if MODE == 'min':
            # slice based-on min_depth
            img = img[:, :, :test_loader.dataset.min_depth]

        img = img.cuda()
        y_true = y_true.cuda()

        if flatten:
            bs, c, depth, h, w = img.shape
            img = img.view(-1, c*depth, h, w)

        with torch.no_grad():
            if rnn:
                img = img.permute(0, 2, 1, 3, 4)

                state = None
                loss = 0.0
                for t in range(img.size(1)):
                    outputs, state = net(img[:, t], state)
                    loss += criterion(outputs, torch.max(y_true, 1)[1])

            else:
                outputs = net(img)
                loss = criterion(outputs, torch.max(y_true, 1)[1])

            _, y_pred = torch.max(outputs, 1)
            _, y_true = torch.max(y_true, 1)

            y_pred_.append(y_pred.detach().cpu().numpy())
            y_true_.append(y_true.cpu().numpy())

            if logging:
                res['y_test'] += y_true.cpu().numpy().tolist()
                res['y_score'] += torch.softmax(outputs,
                                                1)[:, 1].detach().cpu().numpy().tolist()
                res['y_hat'] += y_pred.detach().cpu().numpy().tolist()
                res['CaseIdx'] += list(image_id)

            acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())

            running_loss += loss.item()
            running_acc += acc.item()

    running_loss /= len(test_loader)
    running_acc /= len(test_loader)

    # f1-score
    f1 = f1_score(list(chain(*y_true_)),
                  list(chain(*y_pred_)))

    print(f'[Test] ep : {ep} loss : {running_loss:.4f} \
          acc : {running_acc:.4f} f1 : {f1:.4f}')

    global BEST_SCORE
    if running_acc > BEST_SCORE:
        BEST_SCORE = running_acc
        torch.save(net.state_dict(),
                   f'./history/{net.__class__.__name__}__{MODE}.pt')
        print(f'Save Best Model in HISTORY \n')

    if logging:
        for k, v in res.items():
            res[k] = np.vstack(v)
        return res, running_loss

    else:
        return running_loss
