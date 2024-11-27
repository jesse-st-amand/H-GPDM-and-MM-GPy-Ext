import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from HGPLVM.architectures.base import Architecture_Base
import numpy as np
from GPy import kern
from HGPLVM.hgp_model import HGPLVM
from HGPLVM.GPLVM_node import GPLVM_node as GPNode
from HGPLVM.architectures.base import HGP_arch
import time

class X1_Y1(HGP_arch):
    def __init__(self, HGP, data_set_class, arch_dict = {}, **kwargs):
        if 'top_node_dict' in arch_dict:
            self.top_node_dict = arch_dict['top_node_dict']
        else:
            self.top_node_dict = {}
        temp_top_node_dict = {
                'BC_dict': {
                            'X_init': None,
                            },
                'prior_dict':{},
                'kernel':None,
                'input_dim':None,
                'init':None,
                'opt':None
            }
        super().__init__(HGP, data_set_class, arch_dict, **kwargs)
        self.merge_dicts(temp_top_node_dict, self.top_node_dict)


    def set_up(self, data_set_class):
        super().set_up(data_set_class)
        if self.top_node_dict['attr_dict']['input_dim'] is None:
            raise ValueError('Input dimensions must be set before building this architecture.')
        if self.top_node_dict['attr_dict']['kernel'] is None:
            self.top_node_dict['attr_dict']['kernel'] = (kern.RBF(self.top_node_dict['attr_dict']['input_dim'], 1, ARD=True) +
                                                    kern.Linear(self.top_node_dict['attr_dict']['input_dim'], ARD=True) +
                                                    kern.Bias(self.top_node_dict['attr_dict']['input_dim'], np.exp(-2)))
        if self.top_node_dict['attr_dict']['init'] is None:
            self.top_node_dict['attr_dict']['init'] = 'random'
        if self.top_node_dict['attr_dict']['opt'] is None:
            self.top_node_dict['attr_dict']['opt'] = 'lbfgsb'

        self.top_node_dict['BC_dict']['data_set_class'] = self.data_set_class

        if self.data_set_class.X_init is not None:
            self.top_node_dict['BC_dict']['X_init'] = self.data_set_class.X_init[:,:self.top_node_dict['attr_dict']['input_dim']]
        else:
            self.top_node_dict['BC_dict']['X_init'] = None

        top_node = GPNode(self.Y, self.top_node_dict['attr_dict'], seq_eps=self.data_set_class.seq_eps_train,
                          num_types_seqs=self.data_set_class.num_train_seqs_per_subj_per_act, **self.top_node_dict['attr_dict'])
        top_node.set_backconstraint(BC_dict=self.top_node_dict['BC_dict'])
        top_node.initialize_X(self.top_node_dict['attr_dict']['init'],Y=self.Y,X=self.top_node_dict['BC_dict']['X_init'] )
        self.set_dynamics(self.top_node_dict['prior_dict']['dynamics_dict'])
        top_node.set_prior(prior_dict=self.top_node_dict['prior_dict'], num_seqs=self.data_set_class.sub_num_actions)
        self.GPNode_init_opt(top_node, self.top_node_dict['attr_dict']['opt'], self.top_node_dict['attr_dict']['max_iters'],self.top_node_dict['attr_dict']['GPNode_opt'])
        self.model = HGPLVM(top_node, self)

    def set_Y(self, data_set_class):
        self.Y = data_set_class.Y_train
        self.N, self.D = self.Y.shape

    def IF_setup(self):
        return self.IF_general(self.HGP.arch.dynamics.Y_pred_CCs, self.data_set_class.Y_test_CCs)

    def IF_general(self, prediction, ground_truth):
        IFs = []
        for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
            IFs.append(self.data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
        return IFs

class External_Model_Base(Architecture_Base):
    def __init__(self):
        super().__init__()
        self.pred_classes = []
        self.f1_list = []
        self.score_list = []
        self.msad_list = []
        self.iter_list = []
        self.loss_list = []

    def score(self, init_t=0, seq_len=10, **kwargs):
        self.data_set_class.results_dict = {}

        results = self.data_set_class.get_results(['Y_preds_list', 'pred_trajs', 'pred_traj_lists'],
                                                  self.predict_Ys_Xs, self.data_set_class.Y_test_list, init_t=init_t,
                                                  seq_len=self.wrapper.arch_dict['sample_len'],
                                                  **kwargs)
        results = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in results]
        self.Y_test_CCs = self.data_set_class.Y_test_CCs
        self.Y_pred_CCs = self.data_set_class.Y_pos_list_to_stick_dicts_CCs(results[0])
        true_sequences = self.data_set_class.CC_dict_list_to_CC_array_list_min_PV(self.Y_test_CCs)
        pred_sequences = self.data_set_class.CC_dict_list_to_CC_array_list_min_PV(self.Y_pred_CCs)


        scores = self.data_set_class.score(self.wrapper.arch_dict['sample_len'],self.wrapper.arch_dict['scoring_method'], true_sequences, pred_sequences,
                                              self.data_set_class.action_IDs_test,
                                              self.data_set_class.results_dict['pred_trajs'])

        self.score_list.append(scores['avg_norm_distance'])
        self.f1_list.append(scores['f1'])
        self.msad_list.append(scores['avg_norm_msad'])
        print('')
        print('SCORES: ')
        print('avg_norm_distance: ')
        print(scores['avg_norm_distance'])
        print('f1: ')
        print(scores['f1'])
        print('avg_norm_msad: ')
        print(scores['avg_norm_msad'])
        if scores['f1'] == 0:
            f1_score = 0.01
        else:
            f1_score = scores['f1']
        return scores['avg_norm_distance']

    def store_data(self, epoch, score_rate, num_epochs, loss, start):
        if epoch % score_rate == 0 or epoch == num_epochs:
            if epoch != 0:
                torch.cuda.synchronize()
                end = time.time()
                print(f"Elapsed Time: {end - start}")
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
            self.score()
            self.pred_classes.append([self.data_set_class.results_dict['pred_trajs'],
                                      self.data_set_class.action_IDs_test])
            self.iter_list.append(epoch)
            self.loss_list.append(loss)
            torch.cuda.synchronize()
            start = time.time()
        return start

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VAEMotionModel(nn.Module, External_Model_Base):
    def __init__(self, wrapper, data_set_class, sample_len, input_size, num_classes, hidden_size=256, latent_size=32, subseq_len=10):
        super(VAEMotionModel, self).__init__()
        External_Model_Base.__init__(self)
        self.wrapper = wrapper
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.data_set_class = data_set_class
        self.seq_len = data_set_class.num_tps
        self.sample_len = sample_len

        self.encoder = nn.Sequential(
            nn.Linear(input_size * self.sample_len, hidden_size),  # Adjusted input size
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size * self.seq_len)  # Output full sequence
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('CUDA enabled')
        else:
            self.device = torch.device('cpu')
            print('CUDA not available. CPU enabled')
        self.to(self.device)

        print(f"VAE model parameters: {self.count_parameters()}")

    def encode(self, x):
        x = x.to(self.device)
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, self.seq_len, self.input_size)

    def forward(self, x):
        x = x.to(self.device)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), self.classifier(z), mu, log_var

    def predict_and_generate(self, sequence):
        self.eval()
        with torch.no_grad():
            sequence = sequence.to(self.device)
            recon_batch, classification, _, _ = self(sequence)
            predicted_class = torch.argmax(classification, dim=1)
            return predicted_class, recon_batch

    def loss_function(self, recon_x, x, mu, log_var, classification, label):
        recon_x = recon_x.to(self.device)
        x = x.to(self.device)
        mu = mu.to(self.device)
        log_var = log_var.to(self.device)
        classification = classification.to(self.device)
        label = label.to(self.device)

        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        CE = F.cross_entropy(classification, label, reduction='sum')

        if torch.isnan(MSE) or torch.isinf(MSE):
            print(f"NaN/Inf in MSE. recon_x: {recon_x}, x: {x}")
            MSE = torch.tensor(float('inf'), device=self.device)

        if torch.isnan(KLD) or torch.isinf(KLD):
            print(f"NaN/Inf in KLD. mu: {mu}, log_var: {log_var}")
            KLD = torch.tensor(float('inf'), device=self.device)

        if torch.isnan(CE) or torch.isinf(CE):
            print(f"NaN/Inf in CE. classification: {classification}, label: {label}")
            CE = torch.tensor(float('inf'), device=self.device)

        loss = MSE + KLD + CE

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf in total loss. MSE: {MSE}, KLD: {KLD}, CE: {CE}")
            loss = torch.tensor(float('inf'), device=self.device)

        return loss, MSE, KLD, CE

    def optimize(self, optimizer, num_epochs=100):
        # Prepare full sequences and subsequences
        X_seqs = torch.tensor(np.array(self.data_set_class.Y_train_list), dtype=torch.float32).to(self.device)
        labels = torch.tensor(np.array(self.data_set_class.action_IDs_train), dtype=torch.long).to(self.device)
        X_sub_seqs = X_seqs[:, :self.sample_len, :]  # Extract subsequences
        X_full_seqs = X_seqs  # Full sequences as targets

        print(f"Total number of sequences: {len(X_seqs)}")
        print(f"Shape of X_sub_seqs: {X_sub_seqs.shape}")
        print(f"Shape of X_full_seqs: {X_full_seqs.shape}")
        print(f"Shape of labels: {labels.shape}")

        batch_size = min(32, len(X_seqs))
        print(f"Batch size: {batch_size}")

        score_rate = self.wrapper.arch_dict['score_rate']
        start = time.time()
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            num_batches = 0

            for i in range(0, len(X_sub_seqs), batch_size):
                batch_x = X_sub_seqs[i:i + batch_size]
                batch_target = X_full_seqs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                optimizer.zero_grad()

                recon_batch, classification, mu, log_var = self(batch_x)
                loss, MSE, KLD, CE = self.loss_function(recon_batch, batch_target, mu, log_var, classification, batch_labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            start = self.store_data(epoch, score_rate, num_epochs, avg_loss, start)

    def predict_Ys_Xs(self, Y_test_list, init_t, seq_len=10, **kwargs):
        # Prepare subsequences for testing
        X_seqs = torch.tensor(Y_test_list, dtype=torch.float32).to(self.device)
        X_sub_seqs = X_seqs[:, :seq_len, :]  # Extract subsequences

        Y_preds_list = []
        pred_trajs_list = []

        for x_sub in X_sub_seqs:
            x_sub = x_sub.unsqueeze(0)  # Add batch dimension
            pred_traj, Y_pred = self.predict_and_generate(x_sub)

            Y_pred = Y_pred.squeeze(0).detach().cpu().numpy()

            Y_preds_list.append(Y_pred)
            pred_trajs_list.append(pred_traj.item())  # Store as a single integer

        Y_preds_array = np.array(Y_preds_list)

        Y_pred_denorm_list = self.data_set_class.denorm_trajs(Y_preds_array, self.data_set_class.action_IDs_test)
        return Y_pred_denorm_list, pred_trajs_list, pred_trajs_list

    @staticmethod
    def create_array_with_ones(shape, points):
        arr = np.zeros(shape)
        arr[points] = 1
        return arr

    def IF_setup(self):
        return self.IF_general(self.Y_pred_CCs, self.Y_test_CCs)

    def IF_general(self, prediction, ground_truth):
        IFs = []
        for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
            IFs.append(self.data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
        return IFs

'''class LSTMVAEMotionModel(nn.Module, External_Model_Base):
    def __init__(self, wrapper, data_set_class, sample_len, input_size, num_classes, hidden_size, latent_size, num_layers=1):
        super().__init__()
        self.sample_len = sample_len

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, input_size)

        # Latent to hidden state
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)

    def encode(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder_lstm(packed_x)
        h = h_n[-1]  # Use the last layer's hidden state
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_length):
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # Initialize first input as zeros
        decoder_input = torch.zeros(z.size(0), 1, self.decoder_lstm.input_size, device=z.device)

        outputs = []
        for _ in range(target_length):
            output, (h0, c0) = self.decoder_lstm(decoder_input, (h0, c0))
            output = self.fc_output(output)
            outputs.append(output)
            decoder_input = output  # Use current output as next input

        return torch.cat(outputs, dim=1)

    def forward(self, x, lengths):
        mu, log_var = self.encode(x, lengths)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, log_var

    def predict_and_generate(self, sequence, lengths):
        self.eval()
        with torch.no_grad():
            sequence = sequence.to(self.device)
            recon_batch, classification, _, _ = self(sequence, lengths)
            predicted_class = torch.argmax(classification, dim=1)
            return predicted_class, recon_batch

    def loss_function(self, recon_x, x, mu, log_var, classification, label):
        recon_x = recon_x.to(self.device)
        x = x.to(self.device)
        mu = mu.to(self.device)
        log_var = log_var.to(self.device)
        classification = classification.to(self.device)
        label = label.to(self.device)

        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        CE = F.cross_entropy(classification, label, reduction='sum')

        if torch.isnan(MSE) or torch.isinf(MSE):
            print(f"NaN/Inf in MSE. recon_x: {recon_x}, x: {x}")
            MSE = torch.tensor(float('inf'), device=self.device)

        if torch.isnan(KLD) or torch.isinf(KLD):
            print(f"NaN/Inf in KLD. mu: {mu}, log_var: {log_var}")
            KLD = torch.tensor(float('inf'), device=self.device)

        if torch.isnan(CE) or torch.isinf(CE):
            print(f"NaN/Inf in CE. classification: {classification}, label: {label}")
            CE = torch.tensor(float('inf'), device=self.device)

        loss = MSE + KLD + CE

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf in total loss. MSE: {MSE}, KLD: {KLD}, CE: {CE}")
            loss = torch.tensor(float('inf'), device=self.device)

        return loss, MSE, KLD, CE

    def optimize(self, optimizer, num_epochs=100):
        X_seqs = torch.tensor(np.array(self.data_set_class.Y_train_list), dtype=torch.float32).to(self.device)
        labels = torch.tensor(np.array(self.data_set_class.action_IDs_train), dtype=torch.long).to(self.device)

        batch_size = min(32, len(X_seqs))
        print(f"Total number of sequences: {len(X_seqs)}")
        print(f"Batch size: {batch_size}")

        score_rate = self.wrapper.arch_dict['score_rate']
        start = time.time()
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            num_batches = 0

            # Shuffle data
            perm = torch.randperm(len(X_seqs))
            X_seqs = X_seqs[perm]
            labels = labels[perm]

            for i in range(0, len(X_seqs), batch_size):
                batch_x_full = X_seqs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                # Generate random subsequence lengths for the batch
                max_subseq_len = self.seq_len  # Maximum possible subsequence length
                lengths = torch.randint(1, max_subseq_len + 1, (batch_x_full.size(0),))
                lengths, perm_idx = lengths.sort(0, descending=True)
                batch_x_full = batch_x_full[perm_idx]
                batch_labels = batch_labels[perm_idx]

                # Create subsequences
                batch_x = [seq[:l] for seq, l in zip(batch_x_full, lengths)]
                batch_target = batch_x_full  # Full sequences as targets

                # Pad the sequences
                batch_x_padded = nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
                batch_x_padded = batch_x_padded.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()

                recon_batch, classification, mu, log_var = self(batch_x_padded, lengths)
                loss, MSE, KLD, CE = self.loss_function(recon_batch, batch_target, mu, log_var, classification, batch_labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            start = self.store_data(epoch, score_rate, num_epochs, avg_loss, start)

    def predict_Ys_Xs(self, Y_test_list, init_t, seq_len=10, **kwargs):
        X_seqs = torch.tensor(Y_test_list, dtype=torch.float32).to(self.device)

        Y_preds_list = []
        pred_trajs_list = []

        for x in X_seqs:
            # Use self.sample_len for the subsequence length
            x_sub = x[:self.sample_len].unsqueeze(0)  # Add batch dimension
            lengths = torch.tensor([self.sample_len], dtype=torch.long).to(self.device)

            pred_traj, Y_pred_sub = self.predict_and_generate(x_sub, lengths)

            # Generate full-length sequence based on the subset prediction
            Y_pred_full = torch.zeros_like(x).unsqueeze(0)
            Y_pred_full[:, :self.sample_len, :] = Y_pred_sub[:, :self.sample_len, :]

            for t in range(self.sample_len, x.size(0)):
                next_input = Y_pred_full[:, t - self.sample_len:t, :]
                next_lengths = torch.tensor([self.sample_len], dtype=torch.long).to(self.device)
                _, next_pred = self.predict_and_generate(next_input, next_lengths)
                Y_pred_full[:, t, :] = next_pred[:, -1, :]

            Y_pred = Y_pred_full.squeeze(0).detach().cpu().numpy()

            Y_preds_list.append(Y_pred)
            pred_trajs_list.append(pred_traj.item())

        Y_preds_array = np.array(Y_preds_list)

        Y_pred_denorm_list = self.data_set_class.denorm_trajs(Y_preds_array, self.data_set_class.action_IDs_test)
        return Y_pred_denorm_list, pred_trajs_list, pred_trajs_list

    @staticmethod
    def create_array_with_ones(shape, points):
        arr = np.zeros(shape)
        arr[points] = 1
        return arr

    def IF_setup(self):
        return self.IF_general(self.Y_pred_CCs, self.Y_test_CCs)

    def IF_general(self, prediction, ground_truth):
        IFs = []
        for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
            IFs.append(self.data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
        return IFs
'''
class TransformerMotionModel(nn.Module, External_Model_Base):
    def __init__(self, wrapper, data_set_class, sample_len,input_size, num_classes, hidden_size_multiplier=1, num_layers=2,
                 num_heads=4, dropout=0.1, max_seq_length=100):
        super(TransformerMotionModel, self).__init__()
        External_Model_Base.__init__(self)
        self.wrapper = wrapper
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = num_heads * hidden_size_multiplier
        self.max_seq_length = max_seq_length
        self.sample_len = sample_len
        self.input_projection = nn.Linear(input_size, self.hidden_size)

        # Optionally initialize a [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        config = BertConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=self.hidden_size * 4,  # Typically, intermediate_size is larger
            max_position_embeddings=max_seq_length + 1,  # +1 if adding [CLS] token
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        self.transformer = BertModel(config)

        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.generator = nn.Linear(self.hidden_size, input_size)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('CUDA enabled')
        else:
            self.device = torch.device('cpu')
            print('CUDA not available. CPU enabled')
        self.to(self.device)
        print(f"Model initialized on device: {self.device}")  # Debug line

        print(f"Transformer model parameters: {self.count_parameters()}")

    def count_parameters(self):
        # Count parameters in the transformer
        transformer_params = self.transformer.num_parameters()

        # Count parameters in other parts of the model
        other_params = sum(p.numel() for name, p in self.named_parameters()
                           if 'transformer' not in name and p.requires_grad)

        return transformer_params + other_params

    def forward(self, x, attention_mask=None):
        x = x.to(self.device)
        batch_size = x.size(0)

        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if attention_mask is not None:
            cls_attention = torch.ones((batch_size, 1), device=self.device)
            attention_mask = torch.cat((cls_attention, attention_mask), dim=1)

        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)

        transformer_output = self.transformer(inputs_embeds=x, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = transformer_output.last_hidden_state

        classification_output = self.classifier(hidden_states[:, 0, :])  # [CLS] token
        generated_sequence = self.generator(hidden_states[:, 1:, :])     # Exclude [CLS] token

        return classification_output, generated_sequence

    def predict_and_generate(self, sequence, attention_mask=None):
        self.eval()
        with torch.no_grad():
            classification, generated = self(sequence, attention_mask)
            predicted_class = torch.argmax(classification, dim=1)
            return predicted_class, generated

    def optimize(self, criterion_classification, criterion_generation, optimizer, num_epochs=100):
        X_seqs = torch.tensor(np.array(self.data_set_class.Y_train_list), dtype=torch.float32).to(self.device)
        Y_seqs = torch.tensor(np.array(self.data_set_class.Y_train_list), dtype=torch.float32).to(self.device)
        action_IDs = torch.tensor(self.data_set_class.action_IDs_train, dtype=torch.long).to(self.device)

        # Debug prints to ensure tensors are on the correct device
        #print(f"X_seqs device: {X_seqs.device}, Y_seqs device: {Y_seqs.device}, action_IDs device: {action_IDs.device}")

        score_rate = self.wrapper.arch_dict['score_rate']
        start = time.time()
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # Create attention mask and ensure it's on the correct device
            attention_mask = (X_seqs != 0).any(dim=-1).float().to(self.device)
            #print(f"Attention mask device: {attention_mask.device}")  # Debug line

            classification, generated = self(X_seqs, attention_mask)
            #print(f"Classification device: {classification.device}, Generated device: {generated.device}")  # Debug line

            # Loss calculations
            loss_classification = criterion_classification(classification, action_IDs)
            loss_generation = criterion_generation(generated, Y_seqs)
            loss = loss_classification + loss_generation

            # Debug loss values and device
            #print(f"Loss classification: {loss_classification.item()}, Loss generation: {loss_generation.item()}")
            #print(f"Total loss device: {loss.device}")  # Debug line

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            start = self.store_data(epoch, score_rate, num_epochs, loss.item(), start)


    def predict_Ys_Xs(self, Y_test_list, init_t, seq_len=10, **kwargs):
        X_seqs = torch.tensor(np.array(Y_test_list), dtype=torch.float32).to(self.device)

        # Use only the first self.sample_len time points for prediction
        X_seqs_subset = X_seqs[:, :self.sample_len, :]

        attention_mask = (X_seqs_subset != 0).any(dim=-1).float().to(self.device)

        pred_trajs, Y_preds = self.predict_and_generate(X_seqs_subset, attention_mask)

        # Generate full-length sequences based on the subset predictions
        Y_preds_full = torch.zeros_like(X_seqs)
        Y_preds_full[:, :self.sample_len, :] = Y_preds
        for t in range(self.sample_len, X_seqs.size(1)):
            _, next_pred = self.predict_and_generate(Y_preds_full[:, t - self.sample_len:t, :])
            Y_preds_full[:, t, :] = next_pred[:, -1, :]

        # Move tensors to CPU and convert to numpy
        Y_preds_list = [Y_pred.detach().cpu().numpy() for Y_pred in Y_preds_full]
        pred_trajs = pred_trajs.cpu().numpy()
        pred_trajs_list = [self.create_array_with_ones(self.num_classes, [pred_traj]) for pred_traj in pred_trajs]

        Y_pred_denorm_list = self.data_set_class.denorm_trajs(Y_preds_list, self.data_set_class.action_IDs_test)
        return Y_pred_denorm_list, pred_trajs, pred_trajs_list

    @staticmethod
    def create_array_with_ones(shape, points):
        arr = np.zeros(shape)
        arr[points] = 1
        return arr

    def IF_setup(self):
        return self.IF_general(self.Y_pred_CCs, self.Y_test_CCs)

    def IF_general(self, prediction, ground_truth):
        IFs = []
        for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
            IFs.append(self.data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
        return IFs


class SequenceClassifier(nn.Module, External_Model_Base):
    def __init__(self, wrapper, data_set_class, sample_len,input_size, num_classes, hidden_size=50, num_layers=2,
                 seq_len=100, **kwargs):
        super(SequenceClassifier, self).__init__()
        External_Model_Base.__init__(self)
        self.num_classes = num_classes
        self.data_set_class = data_set_class
        self.wrapper = wrapper
        hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.generator = nn.Linear(hidden_size, input_size)
        self.sample_len = sample_len
        self.output_seq_len = seq_len

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('CUDA enabled')
        else:
            self.device = torch.device('cpu')
            print('CUDA not available. CPU enabled')
        self.to(self.device)

        print(f"LSTM model parameters: {self.count_parameters()}")

    def forward(self, x, predict_length=None):
        x = x.to(self.device)
        batch_size = x.size(0)
        seq_length = x.size(1)

        if predict_length is None:
            predict_length = seq_length

        lstm_out, (hn, cn) = self.lstm(x)
        out_classification = self.classifier(lstm_out[:, -1, :])

        generated_sequence = torch.zeros((batch_size, predict_length, x.size(2)), device=self.device)

        for t in range(predict_length):
            if t < seq_length:
                current_output = self.generator(lstm_out[:, t, :])
            else:
                current_input = current_output.unsqueeze(1)
                lstm_out, (hn, cn) = self.lstm(current_input, (hn, cn))
                current_output = self.generator(lstm_out[:, -1, :])

            generated_sequence[:, t, :] = current_output

        return out_classification, generated_sequence

    def predict_and_generate(self, sequence, predict_length=None):
        self.eval()
        with torch.no_grad():
            sequence = sequence.to(self.device)
            classification, generated = self(sequence, predict_length)
            predicted_class = torch.argmax(classification, dim=1)
            return predicted_class, generated

    def optimize(self, criterion_classification, criterion_generation, optimizer, num_epochs=100):
        num_seqs = len(self.data_set_class.Y_train_list)
        max_seq_len = max(len(seq) for seq in self.data_set_class.Y_train_list)

        # Pre-allocate a numpy array for all sequences
        X_seqs = np.zeros((num_seqs, max_seq_len, self.data_set_class.Y_train.shape[1]))

        # Fill the array with data, padding shorter sequences
        for i, Y_train in enumerate(self.data_set_class.Y_train_list):
            X_seqs[i, :len(Y_train), :] = Y_train

        # Convert the entire numpy array to a tensor at once
        X_seqs_tensor = torch.from_numpy(X_seqs).float().to(self.device)

        # Convert action_IDs to a tensor
        action_IDs = torch.tensor(self.data_set_class.action_IDs_train, dtype=torch.long).to(self.device)

        score_rate = self.wrapper.arch_dict['score_rate']
        start = time.time()
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # Forward pass with full sequences
            classification, generated = self(X_seqs_tensor)

            # Calculate losses
            loss_classification = criterion_classification(classification, action_IDs)
            loss_generation = criterion_generation(generated, X_seqs_tensor)
            loss = loss_classification + loss_generation

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            start = self.store_data(epoch, score_rate, num_epochs, loss.item(), start)

    '''def predict_Ys_Xs(self, Y_test_list, init_t, seq_len=10, **kwargs):
        num_seqs = len(Y_test_list)
        X_seqs = torch.from_numpy(np.array(Y_test_list)).float().to(self.device)

        pred_trajs, Y_preds = self.predict_and_generate(X_seqs)

        # Move tensors to CPU and convert to numpy
        Y_preds_list = [Y_pred.detach().cpu().numpy() for Y_pred in Y_preds]
        pred_trajs = pred_trajs.cpu().numpy()
        pred_trajs_list = [self.create_array_with_ones(self.num_classes, [pred_traj]) for pred_traj in pred_trajs]

        Y_pred_denorm_list = self.data_set_class.denorm_trajs(Y_preds_list, self.data_set_class.action_IDs_test)
        return Y_pred_denorm_list, pred_trajs, pred_trajs_list'''

    def predict_Ys_Xs(self, Y_test_list, init_t, seq_len=10, **kwargs):
        num_seqs = len(Y_test_list)
        X_seqs = torch.from_numpy(np.array(Y_test_list)).float().to(self.device)

        # Use only the first self.sample_len time points for prediction
        X_seqs_subset = X_seqs[:, :self.sample_len, :]

        pred_trajs, Y_preds_subset = self.predict_and_generate(X_seqs_subset)

        # Generate full-length sequences based on the subset predictions
        Y_preds_full = torch.zeros_like(X_seqs)
        Y_preds_full[:, :self.sample_len, :] = Y_preds_subset
        for t in range(self.sample_len, X_seqs.size(1)):
            _, next_pred = self.predict_and_generate(Y_preds_full[:, t - self.sample_len:t, :])
            Y_preds_full[:, t, :] = next_pred[:, -1, :]

        # Move tensors to CPU and convert to numpy
        Y_preds_list = [Y_pred.detach().cpu().numpy() for Y_pred in Y_preds_full]
        pred_trajs = pred_trajs.cpu().numpy()
        pred_trajs_list = [self.create_array_with_ones(self.num_classes, [pred_traj]) for pred_traj in pred_trajs]

        Y_pred_denorm_list = self.data_set_class.denorm_trajs(Y_preds_list, self.data_set_class.action_IDs_test)
        return Y_pred_denorm_list, pred_trajs, pred_trajs_list

    @staticmethod
    def create_array_with_ones(shape, points):
        arr = np.zeros(shape)
        arr[points] = 1
        return arr

    def IF_setup(self):
        return self.IF_general(self.Y_pred_CCs, self.Y_test_CCs)

    def IF_general(self, prediction, ground_truth):
        IFs = []
        for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
            IFs.append(self.data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
        return IFs