import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric, InfoNCE, random_aug, consis_loss


def train(model, data, optimizer, args):
    model.train()
    optimizer.zero_grad()

    output, h = model(data.x, data.edge_index,data.edge_weight)
    preds = (output.squeeze() > 0).type_as(data.y)

    loss = {}
    loss['train'] = F.binary_cross_entropy_with_logits(
        output[data.train_mask], data.y[data.train_mask].unsqueeze(1).float().to(args.device))
    loss['val'] = F.binary_cross_entropy_with_logits(
        output[data.val_mask], data.y[data.val_mask].unsqueeze(1).float().to(args.device), weight=args.val_ratio)

    loss['train'].backward()
    optimizer.step()

    return loss



def evaluate_ged4(x, classifier, MLP_F,encoder, data, args):
    classifier.eval()

    encoder.eval()

    with torch.no_grad():
        h1 = encoder(data.x, data.edge_index, data.adj_norm_sp,data.edge_weight)
        h_F = MLP_F(h1)
        output = classifier(h_F)

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys

def evaluate_ged3(x, classifier, discriminator, encoder, data, args):
    classifier.eval()
    encoder.eval()

    with torch.no_grad():
        h = encoder(data.x, data.edge_index, data.adj_norm_sp,data.edge_weight)
        output = torch.sigmoid(classifier(h))
        # output2 = discriminator(h)

        # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
        #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0.5).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0.5).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys
