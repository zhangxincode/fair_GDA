import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch.sparse


from dataset import *
from model import *
from utils import *
from learn import *
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from data_edit import DE_X,DE_A,train_data_edit,train_data_aug,DataAug







def run(data, args, data2):
    acc, f1, auc_roc, parity, equality = np.zeros([args.runs, len(data2)]), np.zeros([args.runs, len(data2)]), np.zeros(
        [args.runs, len(data2)]), np.zeros([args.runs, len(data2)]), np.zeros([args.runs, len(data2)])

    criterion = nn.BCEWithLogitsLoss()
    data = data.to(args.device)
    for i in range(len(data2)):
        data2[i] = data2[i].to(args.device)
        data2[i].test_mask = data2[i].test_mask | data2[i].val_mask | data2[i].test_mask

    t_idx_s0 = data.sens[data.train_mask] == 0
    t_idx_s1 = data.sens[data.train_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, data.y[data.train_mask] == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, data.y[data.train_mask] == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, data.y[data.train_mask] == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, data.y[data.train_mask] == 0)
    t_num_s0_y1, t_num_s1_y1, t_num_s0_y0, t_num_s1_y0, = sum(t_idx_s0_y1), sum(t_idx_s1_y1), sum(t_idx_s0_y0), sum(
        t_idx_s1_y0),

    idx_s0 = data.sens == 0
    idx_s1 = data.sens == 1
    idx_s0_y1 = torch.logical_and(idx_s0, data.y == 1)
    idx_s1_y1 = torch.logical_and(idx_s1, data.y == 1)
    idx_s0_y0 = torch.logical_and(idx_s0, data.y == 0)
    idx_s1_y0 = torch.logical_and(idx_s1, data.y == 0)
    num_s0_y1, num_s1_y1, num_s0_y0, num_s1_y0, = sum(idx_s0_y1), sum(idx_s1_y1), sum(idx_s0_y0), sum(idx_s1_y0),

    eweight = torch.ones(data.edge_index.shape[1]).to(data.x.device)
    adj = torch.sparse_coo_tensor(data.edge_index, eweight, [data.x.shape[0], data.x.shape[0]])
    A2 = torch.spmm(adj, adj)
    '==========Training============='
    for count in range(args.runs):
        MLP_F = MLP_net(args).to(args.device)
        optimizer_F = torch.optim.Adam(params=MLP_F.parameters(), lr=args.fairnet_lr)

        MLP_B = MLP_net(args).to(args.device)
        optimizer_B = torch.optim.Adam(params=MLP_B.parameters(), lr=args.baisnet_lr)

        discriminator_F = MLP_discriminator(args).to(args.device)
        optimizer_D_F = torch.optim.Adam(params=discriminator_F.parameters(), lr=args.discri_F_lr)

        discriminator_B = MLP_discriminator(args).to(args.device)
        optimizer_D_B = torch.optim.Adam(params=discriminator_B.parameters(), lr=args.discri_B_lr)
        # Note the file path here
        if args.dataset == 'bail':
            train_name = "_B0"
        elif args.dataset == "credit":
            train_name = "_C0"
        elif args.dataset == 'model_para_pokec':
            train_name = "_z"

        classifier = torch.load('./model_para/{}/{}_classifier_best_0.pth'.format(args.dataset, train_name),
                                weights_only=False).to(args.device)  # ,map_location='cpu'
        optimizer_C = torch.optim.Adam(params=classifier.parameters(), lr=args.classify_lr)

        encoder = torch.load('./model_para/{}/{}_encoder_best_0.pth'.format(args.dataset, train_name),
                             weights_only=False).to(args.device)
        optimizer_E = torch.optim.Adam(params=encoder.parameters(), lr=args.encoder_lr)
        data3 = data.clone()
        data_aug = DataAug(classifier, encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1], args)
        de_a = DE_A(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1], args)
        de_x = DE_X(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1], args)
        data1 = update_labels_by_neighbors_with_predictions(data3, encoder, classifier)

        seed_everything(args.seed + count)
        discriminator_B.reset_parameters()
        discriminator_F.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()
        MLP_F.reset_parameters()
        MLP_B.reset_parameters()
        for epoch in range(args.epochs):
            ''' Train discriminator F'''
            encoder.eval()
            MLP_F.train()
            discriminator_F.train()
            for i in range(args.df_epochs):
                optimizer_F.zero_grad()
                optimizer_D_F.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                pred_B = discriminator_F(h_F)
                # The discriminator loss is the sum of two branch losses
                loss_D = criterion(pred_B.view(-1), data1.x[:,data1.sen_idx])
                loss_D.backward()
                optimizer_D_F.step()
                optimizer_F.step()


            '''   Train classifier'''
            encoder.eval()
            MLP_F.train()
            classifier.train()
            for i in range(args.class_epochs):
                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                output_class = classifier(h_F)
                loss_c = criterion(output_class, data1.new_probs[:,1].unsqueeze(1).float())
                loss_c.backward()
                optimizer_C.step()
                optimizer_F.step()





            ''' Adversarial training MLP_F '''
            discriminator_F.eval()
            classifier.eval()
            encoder.eval()
            MLP_F.train()
            for i in range(args.ad_MLP_F_epochs):
                optimizer_F.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                pred_F_adv = torch.sigmoid(discriminator_F(h_F))
                loss_adv_F = criterion(pred_F_adv.view(-1),0.5 * torch.ones_like(pred_F_adv.view(-1)))
                loss_adv_F.backward()
                optimizer_F.step()


            '''Train discriminator B'''
            encoder.eval()
            MLP_B.train()
            discriminator_B.train()
            for i in range(args.db_epochs):
                optimizer_B.zero_grad()
                optimizer_D_B.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_B = MLP_B(h)
                pred_B_adv = discriminator_B(h_B)
                loss_adv_B = criterion(pred_B_adv.view(-1), data1.x[:, data1.sen_idx])
                loss_adv_B.backward()
                optimizer_B.step()
                optimizer_D_B.step()


            '''  Alienation of different classes '''
            encoder.train()
            MLP_F.eval()
            MLP_B.eval()
            classifier.train()
            for i in range(args.align_epochs):
                optimizer_C.zero_grad()
                optimizer_F.zero_grad()
                optimizer_B.zero_grad()
                optimizer_E.zero_grad()
                h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)

                # Classification loss
                h_F = MLP_F(h)
                h_B = MLP_B(h)
                logits = classifier(h_F)
                task_loss = criterion(logits, data1.new_probs[:,1].unsqueeze(1).float())

                # Separation loss
                hF = F.normalize(h_F, dim=1)
                hB = F.normalize(h_B, dim=1)
                cos_sim = (hF * hB).sum(dim=1)
                sigma_max = torch.linalg.svdvals(cos_sim)[0]  # 最大奇异值
                loss_ortho = (cos_sim ** 2).mean() + 0.1 * sigma_max

                # Similarity loss
                h0 = h.clone()
                h0[:, data.sen_idx] = 0

                h1 = h.clone()
                h1[:, data.sen_idx] = 1
                batch_size = h1.shape[0]
                z1 = F.normalize(h0)
                z2 = F.normalize(h1)
                sim = torch.mm(z1, z2.t()) / 0.5  # [batch_size, batch_size]

                # Construct positive sample similarity vector (diagonal)
                pos_sim = sim.diag().view(batch_size, 1)  # [batch_size, 1]

                denom = torch.logsumexp(sim, dim=1, keepdim=True)  # [batch_size, 1]

                # Loss for each anchor: - (positive sample similarity - denominator)
                loss_i = - (pos_sim - denom).mean()

                # Symmetrically, from z2 to z1
                denom_j = torch.logsumexp(sim.t(), dim=1, keepdim=True)
                loss_j = - (pos_sim - denom_j).mean()  # Note, positive sample similarity is still diagonal here, but remains unchanged after transpose

                # Or, recalculate positive samples with z2 as anchor, but note that positive samples are still diagonal, so pos_sim can be used directly

                loss_similar = (loss_i + loss_j) / 2

                #loss = (0.5 * task_loss + 0.2 * loss_ortho ) / (task_loss + loss_ortho )
                # if args.inid == '_B2':
                #     loss = (0.3*task_loss + 0.2 * loss_ortho + 0.5 * loss_similar)/(task_loss + loss_ortho + loss_similar)
                # else: #if args.inid == '_B1':
                loss = (0.5*task_loss + 0.2 * loss_ortho + 0.3 * loss_similar)/(task_loss + loss_ortho + loss_similar) # C2-4 :424
                if args.inid =='_C2' or args.inid =='_C3' or args.inid =='_C4':
                    loss = (0.4*task_loss + 0.2 * loss_ortho + 0.4 * loss_similar)/(task_loss + loss_ortho + loss_similar) # C2-4 :424
                loss.backward()
                optimizer_C.step()
                optimizer_F.step()
                optimizer_B.step()
                optimizer_E.step()


            "=====Testing======="
            encoder.eval()
            discriminator_F.eval()
            classifier.eval()
            MLP_F.eval()
            MLP_B.eval()

            # Evaluation metrics initialization
            test_acc = [0 for n in range(len(data2))]
            best_val_tradeoff = [0 for n in range(len(data2))]
            test_auc_roc = [0 for n in range(len(data2))]
            test_f1 = [0 for n in range(len(data2))]
            test_parity = [0 for n in range(len(data2))]
            test_equality = [0 for n in range(len(data2))]



            for i in range(len(data2)):
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged4(
                    data2[i].x, classifier,MLP_F,encoder, data2[i], args)


                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff[i]:
                    test_acc[i] = accs['test']
                    test_auc_roc[i] = auc_rocs['test']
                    test_f1[i] = F1s['test']
                    test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff[i] = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

            # Data editing training
            if (epoch+1)%args.de_train == 0:
                if args.de_traintype_switch==0:
                    data1 = train_data_edit(data1,de_a,de_x,args)  # xa separate training
                elif args.de_traintype_switch==1:
                    data1 = train_data_aug(data1,data_aug,args) # xa combined training


        for i in range(len(args.strlist)):
            acc[count][i] = test_acc[i]
            f1[count][i] = test_f1[i]
            auc_roc[count][i] = test_auc_roc[i]
            parity[count][i] = test_parity[i]
            equality[count][i] = test_equality[i]



    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,help="Dataset type",default='bail')#german
    parser.add_argument('--inid', type=str, help="Input dataset",default='_B1')
    parser.add_argument('--runs', type=int, help="Number of runs",default=1) # 5

    parser.add_argument('--dataset', type=str,help="Dataset type",default='german')#german
    parser.add_argument('--inid', type=str, help="Input dataset",default='')
    parser.add_argument('--runs', type=int, help="Number of runs",default=5) # 5
    parser.add_argument('--encoder', type=str,help="Encoder type", default='GCN')
    parser.add_argument('--prop', type=str, help="CCN selection",default='scatter')

    parser.add_argument('--hidden', type=int,help="Encoder output feature dimension", default=16)

    parser.add_argument('--seed', type=int,help="Initialization seed",default=22)# B1->14
    parser.add_argument('--gpu', type=int, help="GPU ID to use, automatically switches to CPU if not available",default=0)

    parser.add_argument('--dropout', type=float, help="Encoder dropout probability",default=0.5)
    parser.add_argument('--top_k', type=int,help="Use superman algorithm to calculate top K features most similar to sensitive attributes",default=10)
    parser.add_argument('--alpha', type=float, help="Calculate auc_rocs+F1+acc-args.alpha*(tmp_parity+tmp_equality)",default=1)

    # Learning rate parameters
    parser.add_argument('--fairnet_lr', type=float, help="Fair network learning rate", default=0.001)
    parser.add_argument('--baisnet_lr', type=float, help="Bias network learning rate", default=0.001)
    parser.add_argument('--discri_F_lr', type=float, help="Discriminator F learning rate", default=0.01)
    parser.add_argument('--discri_B_lr', type=float, help="Discriminator B learning rate", default=0.01)
    parser.add_argument('--classify_lr', type=float, help="Classifier learning rate", default=0.01)
    parser.add_argument('--encoder_lr', type=float, help="Encoder learning rate", default=0.01)

    parser.add_argument('--de_feature_lr', type=float, help="Data editing node feature training learning rate", default=0.001)
    parser.add_argument('--de_edge_lr', type=float, help="Data editing edge feature training learning rate", default=0.001)

    # Training epoch parameter adjustments
    parser.add_argument('--epochs', type=int, help="Total epochs for overall model fine-tuning", default=20)
    parser.add_argument('--df_epochs', type=int, help="Total epochs for discriminator F fine-tuning", default=20)
    parser.add_argument('--db_epochs', type=int, help="Total epochs for discriminator B fine-tuning", default=20)
    parser.add_argument('--class_epochs', type=int, help="Total epochs for classifier training fine-tuning", default=40)
    parser.add_argument('--ad_MLP_F_epochs', type=int, help="Total epochs for adversarial training fair network F fine-tuning", default=20)
    parser.add_argument('--align_epochs', type=int, help="Total epochs for fair network and bias network alienation, encoder,classify fine-tuning",default=10)


    # Data editing related parameters

    parser.add_argument('--de_train', type=int, help="How often to train data editing per epoch", default=4)
    parser.add_argument('--de_traintype_switch', type=int, help="Whether to use combined training (1) or separate training (0) for node features and edges",
                        default=0)
    parser.add_argument('--de_together_epochs', type=int, help="Total epochs for combined training in data editing", default=5)

    parser.add_argument('--de_separate_epochs', type=int, help="Total epochs for separate training in data editing", default=5)
    parser.add_argument('--de_separate_node_epochs', type=int, help="Total epochs for node feature training in separate training", default=5)
    parser.add_argument('--de_separate_edge_epochs', type=int, help="Total epochs for edge weight feature training in separate training",
                        default=5)


    args = parser.parse_args()
    args.strlist = None
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    seed_everything(args.seed)

    # Get data
    data, _ , args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.inid, args.top_k)


    # Set test set categories
    data2 = []
    if args.dataset == "credit":

        args.strlist = [args.inid]
        for i in range(len(args.strlist)):
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    elif args.dataset == "bail":
        args.strlist = [args.inid]
        for i in range(len(args.strlist)):
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    elif args.dataset == "model_para_pokec":
        args.strlist = [args.inid]
        args.inidIndex = args.strlist.index(args.inid)
        for i in range(len(args.strlist)):
            if args.inidIndex == i:
                data2.append(data)
                continue
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    args.num_features, args.num_classes = data.x.shape[1], len(data.y.unique()) - 1
    if args.dataset == "model_para_pokec":
        args.num_classes = 1
    args.train_ratio, args.val_ratio = torch.tensor(
        [(data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), \
                                       torch.tensor(
                                           [(data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(args.train_ratio) / args.train_ratio, \
                                       torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[data.y[data.train_mask].long()], \
                                       args.val_ratio[data.y[data.val_mask].long()]



    acc, f1, auc_roc, parity, equality = run(data, args, data2) # data is training data.data2 is testing data
    for i in range(len(args.strlist)):
        print("==========={}============".format(args.inid+args.strlist[i]))
        print('Acc     :{:.5f}'.format( 100*np.mean(acc.T[i])))
        print('auc_roc :{:.5f}'.format( 100*np.mean(auc_roc.T[i])))
        print('F1      :{:.5f}'.format( 100*np.mean(f1.T[i])))
        print('parity  :{:.5f}'.format( 100*np.mean(parity.T[i])))
        print('equality:{:.5f}'.format( 100*np.mean(equality.T[i])))