import torch
import torch.nn as nn
import pickle
from utils import *
from collections import OrderedDict
import pdb

class Model(object):
    def __init__(self, args, my_id_bank):
        self.args = args
        self.my_id_bank = my_id_bank
        self.model = self.prepare_model(args.model_name)


    def prepare_model(self, model_name):
        if self.my_id_bank is None:
            print('ERR: Please load an id_bank before model preparation!')
            return None

        self.config = {'alias': model_name,
              'batch_size': self.args.batch_size, #1024,
              'self_train': self.args.self_train, #1024,
              'optimizer': 'adam',
              'adam_lr': self.args.lr, #0.005, #1e-3,
              'latent_dim': self.args.latent_dim, #hidden_units[2]
              'num_negative': self.args.num_negative, #4
              'l2_regularization': self.args.l2_reg, #1e-07,
              'use_cuda': torch.cuda.is_available() and self.args.cuda, #False,
              'device': self.args.device,
              'embedding_user': None,
              'embedding_item': None,
              'save_trained': True,
              'num_users': int(self.my_id_bank.last_user_index+1),
              'num_items': int(self.my_id_bank.last_item_index+1),
              'num_tasks': len(self.args.src_markets.split('-'))+1,
              'num_shared_experts': int(self.args.num_shared_experts),
              'hidden_units' : [int(i) for i in self.args.hidden_units.split(',')]
        }
        print(self.config)
        if model_name == 'gmf':
            print('Model is GMF++!')
            self.model = GMF(self.config)
        if model_name == 'dnn':
            print('Model is DNN!')
            self.model = DNN(self.config)
        if model_name == 'sb':
            print('Model is SharedBottom!')
            self.model = SharedBottom(self.config)
        if model_name == 'cs':
            print('Model is CrossStitch!')
            self.model = CrossStitch(self.config)
        if model_name == 'mmoe':
            print('Model is MMoE!')
            self.model = MMoE(self.config)
        if model_name == 'cgc':
            print('Model is CGC!')
            self.model = CGC(self.config)
        if model_name == 'adio':
            print('Model is ADIO!')
            self.model = ADIO(self.config)
        if model_name == 'adi':
            print('Model is ADI!')
            self.model = ADI(self.config)
        if model_name == 'adism':
            print('Model is ADISM!')
            self.model = ADISM(self.config)
        self.model = self.model.to(self.args.device)
        print(self.model)
        return self.model


    def fit(self, train_dataloader):
        opt = use_optimizer(self.model, self.config)
        loss_func = nn.CrossEntropyLoss()
        ############
        ## Train
        ############
        self.model.train()
        for epoch in range(self.args.num_epoch):
            print('Epoch {} starts !'.format(epoch))
            total_loss = 0
            # for p in self.model.parameters():
            #     print(p)

            # train the model for some certain iterations
            train_dataloader.refresh_dataloaders()
            data_lens = [len(train_dataloader[idx]) for idx in range(train_dataloader.num_tasks)]
            iteration_num = max(data_lens)
            print(f'iteration num is {iteration_num}')
            for iteration in range(iteration_num):
                for subtask_num in range(train_dataloader.num_tasks): # get one batch from each dataloader
                    cur_train_dataloader = train_dataloader.get_iterator(subtask_num)
                    try:
                        train_user_ids, train_item_ids, train_targets = next(cur_train_dataloader)
                    except:
                        new_train_iterator = iter(train_dataloader[subtask_num])
                        train_user_ids, train_item_ids, train_targets = next(new_train_iterator)
                    train_user_ids = train_user_ids.to(self.args.device)
                    train_item_ids = train_item_ids.to(self.args.device)
                    train_targets = train_targets.to(self.args.device)
                    # print('train_target shape is : {}'.format(train_targets.shape))
                    # print('train_target view -1*20 is : {}'.format(train_targets.view(-1,20)))


                    opt.zero_grad()
                    ratings_pred = self.model(train_user_ids, train_item_ids, subtask_num)

                    loss_st = 0
                    threshold = 0.9
                    ratio_st = (iteration+iteration_num*epoch)/(iteration_num*self.args.num_epoch)
                    ratio_topk_st = ratio_st - 0.4
                    other_tasks = list(range(train_dataloader.num_tasks))
                    other_tasks.remove(subtask_num)
                    if self.config['self_train'] and other_tasks and len(other_tasks)>0 and ratio_st>threshold:
                        for other_task in other_tasks:
                            ratings_pred_st = self.model(train_user_ids, train_item_ids, other_task).view(-1, self.config['num_negative']+1)
                            train_targets_st = train_targets.view(-1, self.config['num_negative']+1)
                            # first win over neg samples
                            _, win_idc = ratings_pred_st.topk(1)
                            ratings_pred_st_1 = ratings_pred_st.masked_select(win_idc.eq(0)).view(-1, self.config['num_negative']+1)
                            train_targets_st_1 = train_targets_st.masked_select(win_idc.eq(0)).view(-1, self.config['num_negative']+1)
                            # second in top ratio
                            if int(ratings_pred_st_1.shape[0]*ratio_topk_st)<1:
                                continue
                            _, topk_idc = ratings_pred_st_1[:, 0].topk(int(ratings_pred_st_1.shape[0]*ratio_topk_st))
                            print(topk_idc.shape)
                            ratings_pred_st_2 = ratings_pred_st_1.gather(0, topk_idc.unsqueeze(1).expand(-1, self.config['num_negative']+1))
                            train_targets_st_2 = train_targets_st_1.gather(0, topk_idc.unsqueeze(1).expand(-1, self.config['num_negative']+1))
                            loss_st = loss_st + self.model.get_loss(ratings_pred_st_2.view(-1), train_targets_st_2.view(-1), self.args.device)

                    loss = self.model.get_loss(ratings_pred, train_targets, self.args.device) + loss_st
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()

            sys.stdout.flush()
            print('-' * 80)

        print('Model is trained! and saved at:')
        self.save()

    # produce the ranking of items for users
    def predict(self, eval_dataloader):
        self.model.eval()
        task_rec_all = []
        task_unq_users = set()
        for test_batch in eval_dataloader:
            test_user_ids, test_item_ids, test_targets = test_batch

            cur_users = [user.item() for user in test_user_ids]
            cur_items = [item.item() for item in test_item_ids]

            test_user_ids = test_user_ids.to(self.args.device)
            test_item_ids = test_item_ids.to(self.args.device)
            test_targets = test_targets.to(self.args.device)
            with torch.no_grad():
                batch_scores = self.model(test_user_ids, test_item_ids, 0)
                batch_scores = batch_scores.detach().cpu().numpy()
            for index in range(len(test_user_ids)):
                task_rec_all.append((cur_users[index], cur_items[index], batch_scores[index][0].item()))

            task_unq_users = task_unq_users.union(set(cur_users))

        task_run_mf = get_run_mf(task_rec_all, task_unq_users, self.my_id_bank)
        return task_run_mf

    ## SAVE the model and idbank
    def save(self):
        if self.config['save_trained']:
            model_dir = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.model'
            cid_filename = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.pickle'
            print(f'--model: {model_dir}')
            print(f'--id_bank: {cid_filename}')
            torch.save(self.model.state_dict(), model_dir)
            with open(cid_filename, 'wb') as centralid_file:
                pickle.dump(self.my_id_bank, centralid_file)

    ## LOAD the model and idbank
    def load(self, checkpoint_dir):
        model_dir = checkpoint_dir
        state_dict = torch.load(model_dir, map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights from {model_dir} are loaded!')

class GMF(nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False
        self.loss_func = nn.BCELoss()

        if config['embedding_user'] is None:
            self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']

        if config['embedding_item'] is None:
            self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']

        self.affine_output = nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = nn.Sigmoid()

    def get_loss(self, ratings_pred, train_targets, device):
        return self.loss_func(ratings_pred.view(-1), train_targets)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class DNNBase(nn.Module):
    def __init__(self, config):
        super(DNNBase, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False
        self.loss_func = nn.CrossEntropyLoss()
        self.num_class = config['num_negative']+1
        self.config = config

        if config['embedding_user'] is None:
            self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']

        if config['embedding_item'] is None:
            self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']

    def get_mlp(self, level, prefix, fc_only=False, use_adi_cat=False):
        if level == 1:
            if fc_only:
                return nn.Sequential(OrderedDict([
                                (f'{prefix}_fc1', nn.Linear(in_features=self.latent_dim, out_features=self.config['hidden_units'][0])),
                               ]))
            else:
                return nn.Sequential(OrderedDict([
                            (f'{prefix}_fc1', nn.Linear(in_features=self.latent_dim, out_features=self.config['hidden_units'][0])),
                            (f'{prefix}_bn1', nn.BatchNorm1d(self.config['hidden_units'][0])),
                            (f'{prefix}_relu1', nn.ReLU()),
                           ]))
        if level == 2:
            if fc_only:
                return nn.Sequential(OrderedDict([
                                (f'{prefix}_fc2', nn.Linear(in_features=self.config['hidden_units'][0], out_features=self.config['hidden_units'][1])),
                               ]))
            else:
                return nn.Sequential(OrderedDict([
                            (f'{prefix}_fc2', nn.Linear(in_features=self.config['hidden_units'][0], out_features=self.config['hidden_units'][1])),
                            (f'{prefix}_bn2', nn.BatchNorm1d(self.config['hidden_units'][1])),
                            (f'{prefix}_relu2', nn.ReLU()),
                           ]))
        if level == 3:
            adi_factor = 3 if use_adi_cat else 1
            if fc_only:
                return nn.Sequential(OrderedDict([
                                (f'{prefix}_fc3', nn.Linear(in_features=self.config['hidden_units'][1]*adi_factor, out_features=self.config['hidden_units'][2]*adi_factor)),
                               ]))
            else:
                return nn.Sequential(OrderedDict([
                            (f'{prefix}_fc3', nn.Linear(in_features=self.config['hidden_units'][1]*adi_factor, out_features=self.config['hidden_units'][2]*adi_factor)),
                            (f'{prefix}_bn3', nn.BatchNorm1d(self.config['hidden_units'][2]*adi_factor)),
                            (f'{prefix}_relu3', nn.ReLU()),
                           ]))
        if level == 4:
            adi_factor = 3 if use_adi_cat else 1
            if fc_only:
                return nn.Sequential(OrderedDict([
                                (f'{prefix}_fc4', nn.Linear(in_features=self.config['hidden_units'][2]*adi_factor, out_features=self.config['hidden_units'][3])),
                               ]))
            else:
                return nn.Sequential(OrderedDict([
                            (f'{prefix}_fc4', nn.Linear(in_features=self.config['hidden_units'][2]*adi_factor, out_features=self.config['hidden_units'][3])),
                            (f'{prefix}_relu4', nn.ReLU()),
                           ]))

    def get_loss(self, ratings_pred, train_targets, device):
        return self.loss_func(ratings_pred.view(-1, self.num_class), torch.zeros(int(ratings_pred.shape[0]/self.num_class), dtype=torch.long, device=device))


    def forward(self, user_indices, item_indices, domain_idc):
        pass
    def init_weight(self):
        pass

class DNN(DNNBase):
    def __init__(self, config):
        super(DNN, self).__init__(config)
        self.umlp1 = self.get_mlp(1, 'u')
        self.umlp2 = self.get_mlp(2, 'u')
        self.umlp3 = self.get_mlp(3, 'u')
        self.umlp4 = self.get_mlp(4, 'u')
        self.imlp1 = self.get_mlp(1, 'i')
        self.imlp2 = self.get_mlp(2, 'i')
        self.imlp3 = self.get_mlp(3, 'i')
        self.imlp4 = self.get_mlp(4, 'i')

        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]
        u = self.umlp1(user_embedding)
        u = self.umlp2(u)
        u = self.umlp3(u)
        u = self.umlp4(u)
        i = self.imlp1(item_embedding)
        i = self.imlp2(i)
        i = self.imlp3(i)
        i = self.imlp4(i)
        logits = torch.sum(torch.mul(u, i), dim=1)
        if self.training:
            logits = logits.view(-1, self.num_class)
            return self.softmax(logits).view(-1, 1)
        else:
            return logits.view(-1, 1)

class SharedBottom(DNNBase):
    def __init__(self, config):
        super(SharedBottom, self).__init__(config)
        self.umlp1 = self.get_mlp(1, 'u')
        self.umlp2 = self.get_mlp(2, 'u')
        self.umlp3 = self.get_mlp(3, 'u')
        #self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u') for i in range(config['num_tasks'])])
        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u') for i in range(config['num_tasks'])])
        self.imlp1 = self.get_mlp(1, 'i')
        self.imlp2 = self.get_mlp(2, 'i')
        self.imlp3 = self.get_mlp(3, 'i')
        #self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i') for i in range(config['num_tasks'])])
        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i') for i in range(config['num_tasks'])])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]
        u = self.umlp1(user_embedding)
        u = self.umlp2(u)
        #u = self.umlp3[domain_idc](u)
        u = self.umlp3(u)
        u = self.umlp4[domain_idc](u)
        i = self.imlp1(item_embedding)
        i = self.imlp2(i)
        i = self.imlp3(i)
        #i = self.imlp3[domain_idc](i)
        i = self.imlp4[domain_idc](i)
        logits = torch.sum(torch.mul(u, i), dim=1)
        if self.training:
            logits = logits.view(-1, self.num_class)
            return self.softmax(logits).view(-1, 1)
        else:
            return logits.view(-1, 1)

class CrossStitch(DNNBase):
    def __init__(self, config):
        super(CrossStitch, self).__init__(config)
        self.umlps1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_tasks'])])
        self.ucs1 = nn.Linear(in_features=config['num_tasks'], out_features=config['num_tasks'])
        self.umlps2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_tasks'])])
        self.ucs2 = nn.Linear(in_features=config['num_tasks'], out_features=config['num_tasks'])
        self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u') for i in range(config['num_tasks'])])
        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u') for i in range(config['num_tasks'])])
        self.imlps1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_tasks'])])
        self.ics1 = nn.Linear(in_features=config['num_tasks'], out_features=config['num_tasks'])
        self.imlps2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_tasks'])])
        self.ics2 = nn.Linear(in_features=config['num_tasks'], out_features=config['num_tasks'])
        self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i') for i in range(config['num_tasks'])])
        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i') for i in range(config['num_tasks'])])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]

        u_arr = [layer(user_embedding) for layer in self.umlps1]
        u = self.ucs1(torch.stack(u_arr, dim=-1))
        u_arr = [layer(u[:, :, idx]) for (idx, layer) in enumerate(self.umlps2)]
        u = self.ucs2(torch.stack(u_arr, dim=-1))[:, :, domain_idc]
        u = self.umlp3[domain_idc](u)
        u = self.umlp4[domain_idc](u)

        i_arr = [layer(item_embedding) for layer in self.imlps1]
        i = self.ucs1(torch.stack(i_arr, dim=-1))
        i_arr = [layer(i[:, :, idx]) for (idx, layer) in enumerate(self.imlps2)]
        i = self.ucs2(torch.stack(i_arr, dim=-1))[:, :, domain_idc]
        i = self.imlp3[domain_idc](i)
        i = self.imlp4[domain_idc](i)

        logits = torch.sum(torch.mul(u, i), dim=1)
        if self.training:
            logits = logits.view(-1, self.num_class)
            return self.softmax(logits).view(-1, 1)
        else:
            return logits.view(-1, 1)

class MMoE(DNNBase):
    def __init__(self, config):
        super(MMoE, self).__init__(config)

        self.umlps1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_shared_experts']+config['num_tasks'])])
        self.umlps2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_shared_experts']+config['num_tasks'])])
        self.ugate = nn.ModuleList([
                         nn.Sequential(OrderedDict([
                            ('u_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts']+config['num_tasks'])),
                            ('u_gate_softmax', nn.Softmax(dim=1)),
                         ]))
                         for i in range(config['num_tasks'])
                     ])
        self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u') for i in range(config['num_tasks'])])
        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u') for i in range(config['num_tasks'])])

        self.imlps1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_shared_experts']+config['num_tasks'])])
        self.imlps2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_shared_experts']+config['num_tasks'])])
        self.igate = nn.ModuleList([
                         nn.Sequential(OrderedDict([
                            ('i_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts']+config['num_tasks'])),
                            ('i_gate_softmax', nn.Softmax(dim=1)),
                         ]))
                         for i in range(config['num_tasks'])
                     ])
        self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i') for i in range(config['num_tasks'])])
        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i') for i in range(config['num_tasks'])])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]


        u_experts = [layer(user_embedding) for layer in self.umlps1]
        u_experts = torch.stack([layer(u_experts[idx]) for (idx, layer) in enumerate(self.umlps2)], dim=-1) # [?*16*3]
        u_gate = torch.stack([layer(user_embedding) for layer in self.ugate], dim=-1) # [?*3*2]
        weighted_u_experts = torch.einsum("abc,acd->abd", (u_experts, u_gate))[:, :, domain_idc]
        u = self.umlp3[domain_idc](weighted_u_experts)
        u = self.umlp4[domain_idc](u)

        i_experts = [layer(item_embedding) for layer in self.imlps1]
        i_experts = torch.stack([layer(i_experts[idx]) for (idx, layer) in enumerate(self.imlps2)], dim=-1) # [?*16*3]
        i_gate = torch.stack([layer(item_embedding) for layer in self.igate], dim=-1) # [?*3*2]
        weighted_i_experts = torch.einsum("abc,acd->abd", (i_experts, i_gate))[:, :, domain_idc]
        i = self.imlp3[domain_idc](weighted_i_experts)
        i = self.imlp4[domain_idc](i)


        logits = torch.sum(torch.mul(u, i), dim=1)
        if self.training:
            logits = logits.view(-1, self.num_class)
            return self.softmax(logits).view(-1, 1)
        else:
            return logits.view(-1, 1)


class CGC(DNNBase):
    def __init__(self, config):
        super(CGC, self).__init__(config)

        self.umlps1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_shared_experts'])])
        self.umlps2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_shared_experts'])])
        self.umlp1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_tasks'])])
        self.umlp2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_tasks'])])
        self.ugate = nn.ModuleList([
                         nn.Sequential(OrderedDict([
                            ('u_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts']+1)),
                            ('u_gate_softmax', nn.Softmax(dim=1)),
                         ]))
                         for i in range(config['num_tasks'])
                     ])
        self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u') for i in range(config['num_tasks'])])
        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u') for i in range(config['num_tasks'])])

        self.imlps1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_shared_experts'])])
        self.imlps2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_shared_experts'])])
        self.imlp1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_tasks'])])
        self.imlp2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_tasks'])])
        self.igate = nn.ModuleList([
                         nn.Sequential(OrderedDict([
                            ('i_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts']+1)),
                            ('i_gate_softmax', nn.Softmax(dim=1)),
                         ]))
                         for i in range(config['num_tasks'])
                     ])
        self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i') for i in range(config['num_tasks'])])
        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i') for i in range(config['num_tasks'])])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]

        u_expert = self.umlp2[domain_idc](self.umlp1[domain_idc](user_embedding))
        u_shared_experts = [layer(user_embedding) for layer in self.umlps1]
        u_experts = torch.stack([u_expert]+[layer(u_shared_experts[idx]) for (idx, layer) in enumerate(self.umlps2)], dim=-1) # [?*16*3]
        u_gate = torch.stack([layer(user_embedding) for layer in self.ugate], dim=-1) # [?*3*2]
        weighted_u_experts = torch.einsum("abc,acd->abd", (u_experts, u_gate))[:, :, domain_idc]
        u = self.umlp3[domain_idc](weighted_u_experts)
        u = self.umlp4[domain_idc](u)

        i_expert = self.imlp2[domain_idc](self.imlp1[domain_idc](item_embedding))
        i_shared_experts = [layer(item_embedding) for layer in self.imlps1]
        i_experts = torch.stack([i_expert]+[layer(i_shared_experts[idx]) for (idx, layer) in enumerate(self.imlps2)], dim=-1) # [?*16*3]
        i_gate = torch.stack([layer(item_embedding) for layer in self.igate], dim=-1) # [?*3*2]
        weighted_i_experts = torch.einsum("abc,acd->abd", (i_experts, i_gate))[:, :, domain_idc]
        i = self.imlp3[domain_idc](weighted_i_experts)
        i = self.imlp4[domain_idc](i)

        logits = torch.sum(torch.mul(u, i), dim=1)
        if self.training:
            logits = logits.view(-1, self.num_class)
            return self.softmax(logits).view(-1, 1)
        else:
            return logits.view(-1, 1)

#class ADIO(DNNBase):
#    def __init__(self, config):
#        super(ADIO, self).__init__(config)
#
#        self.umlps1 = nn.ModuleList([self.get_mlp(1, 'u', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.udsbn1 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][0]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.umlps2 = nn.ModuleList([self.get_mlp(2, 'u', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.udsbn2 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][1]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.umlp1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_tasks'])])
#        self.umlp2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_tasks'])])
#        self.ugate = nn.ModuleList([
#                         nn.Sequential(OrderedDict([
#                            ('u_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts']+1)),
#                            ('u_gate_softmax', nn.Softmax(dim=1)),
#                         ]))
#                         for i in range(config['num_tasks'])
#                     ])
#        self.ugate_adi = nn.Linear(in_features=self.latent_dim, out_features=self.config['num_tasks'])
#        self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u', use_adi_cat=False) for i in range(config['num_tasks'])])
#        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u', use_adi_cat=False) for i in range(config['num_tasks'])])
#
#        self.imlps1 = nn.ModuleList([self.get_mlp(1, 'i', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.idsbn1 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][0]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.imlps2 = nn.ModuleList([self.get_mlp(2, 'i', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.idsbn2 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][1]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.imlp1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_tasks'])])
#        self.imlp2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_tasks'])])
#        self.igate = nn.ModuleList([
#                         nn.Sequential(OrderedDict([
#                            ('i_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts']+1)),
#                            ('i_gate_softmax', nn.Softmax(dim=1)),
#                         ]))
#                         for i in range(config['num_tasks'])
#                     ])
#        self.igate_adi = nn.Linear(in_features=self.latent_dim, out_features=self.config['num_tasks'])
#        self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i', use_adi_cat=False) for i in range(config['num_tasks'])])
#        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i', use_adi_cat=False) for i in range(config['num_tasks'])])
#
#        self.domain_embs = nn.Embedding(config['num_tasks']*2, self.latent_dim, device=config['device'])
#
#        self.relu = nn.ReLU()
#        self.softmax = nn.Softmax(dim=1)
#
#    def forward(self, user_indices, item_indices, domain_idc):
#        if self.trainable_user:
#            user_embedding = self.embedding_user(user_indices)
#        else:
#            user_embedding = self.embedding_user[user_indices]
#        if self.trainable_item:
#            item_embedding = self.embedding_item(item_indices)
#        else:
#            item_embedding = self.embedding_item[item_indices]
#
#        u_expert = self.umlp2[domain_idc](self.umlp1[domain_idc](user_embedding)) # [?*16]
#        u_shared_experts = [self.relu(self.udsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(user_embedding))) for (idx, layer) in enumerate(self.umlps1)]
#        u_experts = torch.stack([u_expert]+[self.relu(self.udsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(u_shared_experts[idx]))) for (idx, layer) in enumerate(self.umlps2)], dim=-1)
#        u_gate = torch.stack([layer(user_embedding) for layer in self.ugate], dim=-1) # [?*3*2]
#        weighted_u_expert = torch.einsum("abc,acd->abd", (u_experts, u_gate))[:,:,domain_idc] # [?*16]
#        u = self.umlp3[domain_idc](weighted_u_expert)
#        u = self.umlp4[domain_idc](u)
#       # u_expert = self.umlp2[domain_idc](self.umlp1[domain_idc](user_embedding)) # [?*16]
#       # u_shared_experts = [self.relu(self.udsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(user_embedding))) for (idx, layer) in enumerate(self.umlps1)]
#       # u_experts = torch.stack([self.relu(self.udsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(u_shared_experts[idx]))) for (idx, layer) in enumerate(self.umlps2)], dim=-1) # [?*16*3]
#       # u_gate = torch.stack([layer(user_embedding) for layer in self.ugate], dim=-1) # [?*3*2]
#       # weighted_u_expert = torch.einsum("abc,acd->abd", (u_experts, u_gate))[:,:,domain_idc] # [?*16]
#       # u_gate_adi_1 = torch.cat([user_embedding, self.domain_embs(torch.LongTensor([domain_idc]*weighted_u_expert.shape[0]).to(self.config['device']))], 1) # [?*2]
#       # u_gate_adi = torch.sigmoid(self.ugate_adi(u_gate_adi_1)) # [?*2]
#       # u_adi_spec = u_expert*u_gate_adi[:,1:2].expand(-1, 16)
#       # u_adi_share = weighted_u_expert*u_gate_adi[:,0:1].expand(-1, 16)
#       # u_adi = torch.cat([u_adi_spec, u_adi_spec*u_adi_share, u_adi_share], 1) # [?*48]
#       # u = self.umlp3[domain_idc](u_adi)
#       # u = self.umlp4[domain_idc](u)
#
#
#        i_expert = self.imlp2[domain_idc](self.imlp1[domain_idc](item_embedding))
#        i_shared_experts = [self.relu(self.idsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(item_embedding))) for (idx, layer) in enumerate(self.imlps1)]
#        i_experts = torch.stack([i_expert]+[self.relu(self.idsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(i_shared_experts[idx]))) for (idx, layer) in enumerate(self.imlps2)], dim=-1)
#        i_gate = torch.stack([layer(item_embedding) for layer in self.igate], dim=-1) # [?*3*2]
#        weighted_i_expert = torch.einsum("abc,acd->abd", (i_experts, i_gate))[:,:,domain_idc] # [?*16]
#        i = self.imlp3[domain_idc](weighted_i_expert)
#        i = self.imlp4[domain_idc](i)
#       # i_expert = self.imlp2[domain_idc](self.imlp1[domain_idc](item_embedding))
#       # i_shared_experts = [self.relu(self.idsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(item_embedding))) for (idx, layer) in enumerate(self.imlps1)]
#       # i_experts = torch.stack([self.relu(self.idsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(i_shared_experts[idx]))) for (idx, layer) in enumerate(self.imlps2)], dim=-1) # [?*16*3]
#       # i_gate = torch.stack([layer(item_embedding) for layer in self.igate], dim=-1) # [?*3*2]
#       # weighted_i_expert = torch.einsum("abc,acd->abd", (i_experts, i_gate))[:,:,domain_idc] # [?*16]
#       # i_gate_adi_1 = torch.cat([item_embedding, self.domain_embs(torch.LongTensor([domain_idc+self.config['num_tasks']]*weighted_i_expert.shape[0]).to(self.config['device']))], 1) # [?*2]
#       # i_gate_adi = torch.sigmoid(self.igate_adi(i_gate_adi_1)) # [?*2]
#       # i_adi_spec = i_expert*i_gate_adi[:,1:2].expand(-1, 16)
#       # i_adi_share = weighted_i_expert*i_gate_adi[:,0:1].expand(-1, 16)
#       # i_adi = torch.cat([i_adi_spec, i_adi_spec*i_adi_share, i_adi_share], 1) # [?*48]
#       # i = self.imlp3[domain_idc](i_adi)
#       # i = self.imlp4[domain_idc](i)
#
#        logits = torch.sum(torch.mul(u, i), dim=1)
#        if self.training:
#            logits = logits.view(-1, self.num_class)
#            return self.softmax(logits).view(-1, 1)
#        else:
#            return logits.view(-1, 1)

class ADI(DNNBase):
    def __init__(self, config):
        super(ADI, self).__init__(config)

        self.umlps1 = nn.ModuleList([self.get_mlp(1, 'u', fc_only=True) for i in range(config['num_shared_experts'])])
        self.udsbn1 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][0]) for i in range(config['num_tasks']*config['num_shared_experts'])])
        self.umlps2 = nn.ModuleList([self.get_mlp(2, 'u', fc_only=True) for i in range(config['num_shared_experts'])])
        self.udsbn2 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][1]) for i in range(config['num_tasks']*config['num_shared_experts'])])
        self.umlp1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_tasks'])])
        self.umlp2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_tasks'])])
        self.ugate = nn.ModuleList([
                         nn.Sequential(OrderedDict([
                            ('u_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts'])),
                            ('u_gate_softmax', nn.Softmax(dim=1)),
                         ]))
                         for i in range(config['num_tasks'])
                     ])
        self.ugate_adi = nn.Linear(in_features=self.latent_dim, out_features=2)
        self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u', use_adi_cat=True) for i in range(config['num_tasks'])])
        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u', use_adi_cat=True) for i in range(config['num_tasks'])])

        self.imlps1 = nn.ModuleList([self.get_mlp(1, 'i', fc_only=True) for i in range(config['num_shared_experts'])])
        self.idsbn1 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][0]) for i in range(config['num_tasks']*config['num_shared_experts'])])
        self.imlps2 = nn.ModuleList([self.get_mlp(2, 'i', fc_only=True) for i in range(config['num_shared_experts'])])
        self.idsbn2 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][1]) for i in range(config['num_tasks']*config['num_shared_experts'])])
        self.imlp1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_tasks'])])
        self.imlp2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_tasks'])])
        self.igate = nn.ModuleList([
                         nn.Sequential(OrderedDict([
                            ('i_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts'])),
                            ('i_gate_softmax', nn.Softmax(dim=1)),
                         ]))
                         for i in range(config['num_tasks'])
                     ])
        self.igate_adi = nn.Linear(in_features=self.latent_dim, out_features=2)
        self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i', use_adi_cat=True) for i in range(config['num_tasks'])])
        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i', use_adi_cat=True) for i in range(config['num_tasks'])])

        self.domain_embs = nn.Embedding(config['num_tasks']*2, self.latent_dim, device=config['device'])

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_indices, item_indices, domain_idc):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]

        u_expert = self.umlp2[domain_idc](self.umlp1[domain_idc](user_embedding)) # [?*16]
        u_shared_experts = [self.relu(self.udsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(user_embedding))) for (idx, layer) in enumerate(self.umlps1)]
        u_experts = torch.stack([self.relu(self.udsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(u_shared_experts[idx]))) for (idx, layer) in enumerate(self.umlps2)], dim=-1) # [?*16*3]
        u_gate = torch.stack([layer(user_embedding) for layer in self.ugate], dim=-1) # [?*3*2]
        weighted_u_expert = torch.einsum("abc,acd->abd", (u_experts, u_gate))[:,:,domain_idc] # [?*16]
        u_gate_adi_1 = user_embedding + self.domain_embs(torch.LongTensor([domain_idc]*weighted_u_expert.shape[0]).to(self.config['device'])) # [?*2]
        u_gate_adi = torch.sigmoid(self.ugate_adi(u_gate_adi_1)) # [?*2]
        u_adi_spec = u_expert*u_gate_adi[:,1:2].expand(-1, u_expert.shape[1])
        u_adi_share = weighted_u_expert*u_gate_adi[:,0:1].expand(-1, weighted_u_expert.shape[1])
        u_adi = torch.cat([u_adi_spec, u_adi_spec*u_adi_share, u_adi_share], 1) # [?*48]
        u = self.umlp3[domain_idc](u_adi)
        u = self.umlp4[domain_idc](u)

        i_expert = self.imlp2[domain_idc](self.imlp1[domain_idc](item_embedding))
        i_shared_experts = [self.relu(self.idsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(item_embedding))) for (idx, layer) in enumerate(self.imlps1)]
        i_experts = torch.stack([self.relu(self.idsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(i_shared_experts[idx]))) for (idx, layer) in enumerate(self.imlps2)], dim=-1) # [?*16*3]
        i_gate = torch.stack([layer(item_embedding) for layer in self.igate], dim=-1) # [?*3*2]
        weighted_i_expert = torch.einsum("abc,acd->abd", (i_experts, i_gate))[:,:,domain_idc] # [?*16]
        i_gate_adi_1 = item_embedding + self.domain_embs(torch.LongTensor([domain_idc+self.config['num_tasks']]*weighted_i_expert.shape[0]).to(self.config['device'])) # [?*2]
        i_gate_adi = torch.sigmoid(self.igate_adi(i_gate_adi_1)) # [?*2]
        i_adi_spec = i_expert*i_gate_adi[:,1:2].expand(-1, i_expert.shape[1])
        i_adi_share = weighted_i_expert*i_gate_adi[:,0:1].expand(-1, weighted_i_expert.shape[1])
        i_adi = torch.cat([i_adi_spec, i_adi_spec*i_adi_share, i_adi_share], 1) # [?*48]
        i = self.imlp3[domain_idc](i_adi)
        i = self.imlp4[domain_idc](i)

        logits = torch.sum(torch.mul(u, i), dim=1)
        if self.training:
            logits = logits.view(-1, self.num_class)
            return self.softmax(logits).view(-1, 1)
        else:
            return logits.view(-1, 1)

#class ADISM(DNNBase):
#    def __init__(self, config):
#        super(ADISM, self).__init__(config)
#
#        self.umlps1 = nn.ModuleList([self.get_mlp(1, 'u', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.udsbn1 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][0]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.umlps2 = nn.ModuleList([self.get_mlp(2, 'u', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.udsbn2 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][1]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.umlp1 = nn.ModuleList([self.get_mlp(1, 'u') for i in range(config['num_tasks'])])
#        self.umlp2 = nn.ModuleList([self.get_mlp(2, 'u') for i in range(config['num_tasks'])])
#        self.ugate = nn.ModuleList([
#                         nn.Sequential(OrderedDict([
#                            ('u_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts'])),
#                            ('u_gate_softmax', nn.Softmax(dim=1)),
#                         ]))
#                         for i in range(config['num_tasks'])
#                     ])
#        self.ugate_adi = nn.Linear(in_features=self.latent_dim, out_features=self.config['num_tasks'])
#        self.umlp3 = nn.ModuleList([self.get_mlp(3, 'u', use_adi_cat=False) for i in range(config['num_tasks'])])
#        self.umlp4 = nn.ModuleList([self.get_mlp(4, 'u', use_adi_cat=False) for i in range(config['num_tasks'])])
#
#        self.imlps1 = nn.ModuleList([self.get_mlp(1, 'i', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.idsbn1 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][0]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.imlps2 = nn.ModuleList([self.get_mlp(2, 'i', fc_only=True) for i in range(config['num_shared_experts'])])
#        self.idsbn2 = nn.ModuleList([nn.BatchNorm1d(self.config['hidden_units'][1]) for i in range(config['num_tasks']*config['num_shared_experts'])])
#        self.imlp1 = nn.ModuleList([self.get_mlp(1, 'i') for i in range(config['num_tasks'])])
#        self.imlp2 = nn.ModuleList([self.get_mlp(2, 'i') for i in range(config['num_tasks'])])
#        self.igate = nn.ModuleList([
#                         nn.Sequential(OrderedDict([
#                            ('i_gate', nn.Linear(in_features=self.latent_dim, out_features=self.config['num_shared_experts'])),
#                            ('i_gate_softmax', nn.Softmax(dim=1)),
#                         ]))
#                         for i in range(config['num_tasks'])
#                     ])
#        self.igate_adi = nn.Linear(in_features=self.latent_dim, out_features=self.config['num_tasks'])
#        self.imlp3 = nn.ModuleList([self.get_mlp(3, 'i', use_adi_cat=False) for i in range(config['num_tasks'])])
#        self.imlp4 = nn.ModuleList([self.get_mlp(4, 'i', use_adi_cat=False) for i in range(config['num_tasks'])])
#
#        self.domain_embs = nn.Embedding(config['num_tasks']*2, self.latent_dim, device=config['device'])
#
#        self.relu = nn.ReLU()
#        self.softmax = nn.Softmax(dim=1)
#
#    def forward(self, user_indices, item_indices, domain_idc):
#        if self.trainable_user:
#            user_embedding = self.embedding_user(user_indices)
#        else:
#            user_embedding = self.embedding_user[user_indices]
#        if self.trainable_item:
#            item_embedding = self.embedding_item(item_indices)
#        else:
#            item_embedding = self.embedding_item[item_indices]
#
#        u_expert = self.umlp2[domain_idc](self.umlp1[domain_idc](user_embedding)) # [?*16]
#        u_shared_experts = [self.relu(self.udsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(user_embedding))) for (idx, layer) in enumerate(self.umlps1)]
#        u_experts = torch.stack([self.relu(self.udsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(u_shared_experts[idx]))) for (idx, layer) in enumerate(self.umlps2)], dim=-1) # [?*16*3]
#        u_gate = torch.stack([layer(user_embedding) for layer in self.ugate], dim=-1) # [?*3*2]
#        weighted_u_expert = torch.einsum("abc,acd->abd", (u_experts, u_gate))[:,:,domain_idc] # [?*16]
#        u_gate_adi_1 = user_embedding + self.domain_embs(torch.LongTensor([domain_idc]*weighted_u_expert.shape[0]).to(self.config['device'])) # [?*2]
#        u_gate_adi = torch.sigmoid(self.ugate_adi(u_gate_adi_1)) # [?*2]
#        u_adi_spec = u_expert*u_gate_adi[:,1:2].expand(-1, u_expert.shape[1])
#        u_adi_share = weighted_u_expert*u_gate_adi[:,0:1].expand(-1, weighted_u_expert.shape[1])
#        u_adi = u_adi_spec + u_adi_share # [?*48]
#        u = self.umlp3[domain_idc](u_adi)
#        u = self.umlp4[domain_idc](u)
#
#        i_expert = self.imlp2[domain_idc](self.imlp1[domain_idc](item_embedding))
#        i_shared_experts = [self.relu(self.idsbn1[domain_idc*self.config['num_shared_experts']+idx](layer(item_embedding))) for (idx, layer) in enumerate(self.imlps1)]
#        i_experts = torch.stack([self.relu(self.idsbn2[domain_idc*self.config['num_shared_experts']+idx](layer(i_shared_experts[idx]))) for (idx, layer) in enumerate(self.imlps2)], dim=-1) # [?*16*3]
#        i_gate = torch.stack([layer(item_embedding) for layer in self.igate], dim=-1) # [?*3*2]
#        weighted_i_expert = torch.einsum("abc,acd->abd", (i_experts, i_gate))[:,:,domain_idc] # [?*16]
#        i_gate_adi_1 = item_embedding + self.domain_embs(torch.LongTensor([domain_idc+self.config['num_tasks']]*weighted_i_expert.shape[0]).to(self.config['device'])) # [?*2]
#        i_gate_adi = torch.sigmoid(self.igate_adi(i_gate_adi_1)) # [?*2]
#        i_adi_spec = i_expert*i_gate_adi[:,1:2].expand(-1, i_expert.shape[1])
#        i_adi_share = weighted_i_expert*i_gate_adi[:,0:1].expand(-1, weighted_i_expert.shape[1])
#        i_adi = i_adi_spec + i_adi_share # [?*48]
#        i = self.imlp3[domain_idc](i_adi)
#        i = self.imlp4[domain_idc](i)
#
#        logits = torch.sum(torch.mul(u, i), dim=1)
#        if self.training:
#            logits = logits.view(-1, self.num_class)
#            return self.softmax(logits).view(-1, 1)
#        else:
#            return logits.view(-1, 1)
