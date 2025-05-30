import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

import numpy as np
from GPy import kern
from HGPLVM.hgp_model import HGPLVM
from HGPLVM.GPLVM_node import GPLVM_node as GPNode
#from HGPLVM.architectures.base import Architecture_Base
from HGPLVM.architectures.base import HGP_arch
import time
from colorama import init
from termcolor import colored

# Initialize colorama for cross-platform colored terminal text
init()

class Architecture_Base():
    def __init__(self, arch_dict = {}, **kwargs):
        if 'attr_dict' in arch_dict:
            self.attr_dict = arch_dict['attr_dict']
        else:
            self.attr_dict = {}

    def get_results(self, Y_list, label, **kwargs):
        results = self.data_set_class.get_results(['Y_preds_list_'+label, 'pred_trajs_'+label, 'pred_traj_lists_'+label],
                                                  self.predict_Ys_Xs, Y_list, init_t=self.wrapper.arch_dict['init_t'],
                                                  seq_len=self.wrapper.arch_dict['sample_len'],
                                                  **kwargs)
        return [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in results]
    
    def get_scores(self,true_sequences, pred_sequences, label, action_IDs_true, **kwargs):
        return self.data_set_class.score(self.wrapper.arch_dict['sample_len'],
                                         self.wrapper.arch_dict['scoring_method'], 
                                         true_sequences, pred_sequences,
                                         action_IDs_true,
                                         self.data_set_class.results_dict['pred_trajs_'+label])
    
    def get_class_preds(self,label,action_IDs):
        return [self.data_set_class.results_dict['pred_trajs_'+label],
                                      action_IDs]
    
    def store_data(self, epoch, score_rate, num_epochs, loss, Y_true_list, Y_true_CCs, action_IDs_true, label):
        if score_rate == 0:
            pass
        elif epoch % score_rate == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
            self.score(Y_true_list, Y_true_CCs, label, epoch, loss, action_IDs_true)
            #torch.cuda.synchronize()
    
    def score(self):
        raise NotImplementedError

class Evaluation_Base():
    def __init__(self, **kwargs):
        self.pred_classes = []
        self.f1_list = []
        self.score_list = []
        self.smoothness_list = []
        self.freeze_list = []
        self.iter_list = []
        self.loss_list = []
        self.label_list = []
        self.results_dict = {}
        

    def scores_append_and_print(self, scores, epoch, loss, label, action_IDs):
        self.pred_classes.append(self.get_class_preds(label, action_IDs))
        self.score_list.append(scores['avg_norm_distance'])
        self.f1_list.append(scores['f1'])
        self.smoothness_list.append(scores['avg_smoothness'])
        self.freeze_list.append(scores['avg_freeze'])
        self.iter_list.append(epoch)
        self.loss_list.append(loss)
        self.label_list.append(label)

        print('')
        print('SCORES: ')
        print('avg_norm_distance: ')
        print(scores['avg_norm_distance'])
        print('f1: ')
        print(scores['f1'])
        print('avg_smoothness: ')
        print(scores['avg_smoothness'])
        print('avg_freeze: ')
        print(scores['avg_freeze'])
        
        if scores['f1'] == 0:
            f1_score = 0.01
        else:
            f1_score = scores['f1']

    def score(self, Y_true_list, Y_true_CCs, label, iters, loss, action_IDs, **kwargs):
        self.data_set_class.results_dict = {}
        results = self.get_results(Y_true_list, label)
        self.results_dict['Y_pred_CCs_'+label] = self.data_set_class.Y_pos_list_to_stick_dicts_CCs(results[0])
        true_sequences = self.data_set_class.CC_dict_list_to_CC_array_list_min_PV(Y_true_CCs)
        pred_sequences = self.data_set_class.CC_dict_list_to_CC_array_list_min_PV(self.results_dict['Y_pred_CCs_'+label])
        scores = self.get_scores(true_sequences, pred_sequences, label, action_IDs)
        self.scores_append_and_print(scores, iters, loss, label, action_IDs)  
        return scores['avg_norm_distance']
    
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class X1_Y1(HGP_arch, Evaluation_Base):
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
        Evaluation_Base.__init__(self)
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
    
class X1_H1_Y1(HGP_arch, Evaluation_Base):
    def __init__(self, HGP, data_set_class, arch_dict = {}, **kwargs):
        if 'top_node_dict' in arch_dict:
            self.top_node_dict = arch_dict['top_node_dict']
        else:
            self.top_node_dict = {}
        if 'H1_node_dict' in arch_dict:
            self.H1_node_dict = arch_dict['H1_node_dict']
        else:
            self.H1_node_dict = {}

        temp_top_node_dict = {
            'BC_dict': {
                'X_init': None,
            },
            'prior_dict': {},
            'kernel': None,
            'input_dim': None,
            'init': None,
            'opt': None
        }

        temp_H1_node_dict = {
            'BC_dict': {
                'X_init': None,
            },
            'prior_dict': {},
            'kernel': None,
            'input_dim': None,
            'init': None,
            'opt': None
        }

        super().__init__(HGP, data_set_class, arch_dict, **kwargs)
        Evaluation_Base.__init__(self)
        self.merge_dicts(temp_top_node_dict, self.top_node_dict)
        self.merge_dicts(temp_H1_node_dict, self.H1_node_dict)

    def set_up(self, data_set_class):
        super().set_up(data_set_class)
        # Validate input dimensions
        if self.top_node_dict['attr_dict']['input_dim'] is None or self.H1_node_dict['attr_dict']['input_dim'] is None:
            raise ValueError('Input dimensions must be set before building this architecture.')
        else:
            print(colored("WARNING: Input dimensions of bottom layer = self.top_node_dict['attr_dict']['input_dim'] + self.H1_node_dict['attr_dict']['input_dim']", 'red'))
            self.H1_node_dict['attr_dict']['input_dim'] = self.top_node_dict['attr_dict']['input_dim'] + self.H1_node_dict['attr_dict']['input_dim']

        if self.top_node_dict['BC_dict']['geo params'] < 1:
            print(colored("WARNING: Geometric parameters of top layer = self.top_node_dict['BC_dict']['geo params'] * self.top_node_dict['attr_dict']['input_dim']", 'red'))
            self.top_node_dict['BC_dict']['geo params'] = int(self.top_node_dict['BC_dict']['geo params'] * self.top_node_dict['attr_dict']['input_dim'])

        # Set up kernels if not provided
        if self.H1_node_dict['attr_dict']['kernel'] is None:
            self.H1_node_dict['attr_dict']['kernel'] = (kern.RBF(self.H1_node_dict['attr_dict']['input_dim'], 1, ARD=True) +
                                                    kern.Linear(self.H1_node_dict['attr_dict']['input_dim'], ARD=True) +
                                                    kern.Bias(self.H1_node_dict['attr_dict']['input_dim'], np.exp(-2)))
        if self.top_node_dict['attr_dict']['kernel'] is None:
            self.top_node_dict['attr_dict']['kernel'] = (kern.RBF(self.top_node_dict['attr_dict']['input_dim'], 1, ARD=True) +
                                                    kern.Linear(self.top_node_dict['attr_dict']['input_dim'], ARD=True) +
                                                    kern.Bias(self.top_node_dict['attr_dict']['input_dim'], np.exp(-2)))

        # Set default initializations if not provided
        if self.H1_node_dict['attr_dict']['init'] is None:
            self.H1_node_dict['attr_dict']['init'] = 'random'
        if self.top_node_dict['attr_dict']['init'] is None:
            self.top_node_dict['attr_dict']['init'] = 'random'

        # Set default optimizers if not provided
        if self.H1_node_dict['attr_dict']['opt'] is None:
            self.H1_node_dict['attr_dict']['opt'] = 'lbfgsb'
        if self.top_node_dict['attr_dict']['opt'] is None:
            self.top_node_dict['attr_dict']['opt'] = 'lbfgsb'

        # Set up data set class references
        self.H1_node_dict['BC_dict']['data_set_class'] = self.data_set_class
        self.top_node_dict['BC_dict']['data_set_class'] = self.data_set_class

        if self.data_set_class.X_init is not None:
            self.top_node_dict['BC_dict']['X_init'] = self.data_set_class.X_init[:,:self.top_node_dict['attr_dict']['input_dim']]
            self.H1_node_dict['BC_dict']['X_init'] = self.data_set_class.X_init[:,:self.H1_node_dict['attr_dict']['input_dim']]
        else:
            self.top_node_dict['BC_dict']['X_init'] = None
            self.H1_node_dict['BC_dict']['X_init'] = None

        # Initialize H1 node
        H_Y = GPNode(self.Y, self.H1_node_dict['attr_dict'], seq_eps=self.data_set_class.seq_eps_train,
                     num_types_seqs=self.data_set_class.num_train_seqs_per_subj_per_act, **self.H1_node_dict['attr_dict'])
        H_Y.set_backconstraint(BC_dict=self.H1_node_dict['BC_dict'])
        H_Y.initialize_X(self.H1_node_dict['attr_dict']['init'], Y=self.Y, X=self.H1_node_dict['BC_dict']['X_init'])
        #self.set_dynamics(self.H1_node_dict['prior_dict']['dynamics_dict'])
        #H_Y.set_prior(prior_dict=self.H1_node_dict['prior_dict'], num_seqs=self.data_set_class.sub_num_actions)
        #self.GPNode_init_opt(H_Y, self.H1_node_dict['attr_dict']['opt'], self.H1_node_dict['attr_dict']['max_iters'], self.H1_node_dict['attr_dict']['GPNode_opt'])

        # Initialize top node
        top_node = GPNode(H_Y.X.values, self.top_node_dict['attr_dict'], seq_eps=self.data_set_class.seq_eps_train,
                         num_types_seqs=self.data_set_class.num_train_seqs_per_subj_per_act, **self.top_node_dict['attr_dict'])
        top_node.SetChild(0, H_Y)
        top_node.set_backconstraint(BC_dict=self.top_node_dict['BC_dict'])
        top_node.initialize_X(self.top_node_dict['attr_dict']['init'], Y=H_Y.X.values, X=self.top_node_dict['BC_dict']['X_init'])
        self.set_dynamics(self.top_node_dict['prior_dict']['dynamics_dict'])
        top_node.set_prior(prior_dict=self.top_node_dict['prior_dict'], num_seqs=self.data_set_class.sub_num_actions)
        #self.GPNode_init_opt(top_node, self.top_node_dict['attr_dict']['opt'], self.top_node_dict['attr_dict']['max_iters'], self.top_node_dict['attr_dict']['GPNode_opt'])

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

class VAEMotionModel(nn.Module, Evaluation_Base, Architecture_Base):
    def __init__(self, wrapper, data_set_class, sample_len, input_size, num_classes, hidden_size=256, latent_size=32, subseq_len=10):
        super(VAEMotionModel, self).__init__()
        Evaluation_Base.__init__(self)
        Architecture_Base.__init__(self)
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
            self.store_data(epoch, score_rate, num_epochs, avg_loss, self.data_set_class.Y_validation_list, self.data_set_class.Y_validation_CCs, self.data_set_class.action_IDs_validation,'validation')
            self.store_data(epoch, score_rate, num_epochs, avg_loss, self.data_set_class.Y_test_list, self.data_set_class.Y_test_CCs, self.data_set_class.action_IDs_test,'test')
        return self.score(self.data_set_class.Y_test_list, self.data_set_class.Y_test_CCs, 'test', num_epochs, avg_loss, self.data_set_class.action_IDs_test,)

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

    def get_latent_space(self, data_list=None):
        """
        Extract latent space representations from the VAE encoder
        
        Parameters:
        -----------
        data_list : list, optional
            List of data sequences to encode. If None, uses training data.
            
        Returns:
        --------
        latent_space : numpy.ndarray
            Matrix of latent representations (N x latent_size)
        """
        self.eval()
        with torch.no_grad():
            if data_list is None:
                # Use training data by default
                X_seqs = torch.tensor(np.array(self.data_set_class.Y_train_list), dtype=torch.float32)
            else:
                X_seqs = torch.tensor(np.array(data_list), dtype=torch.float32)
            
            # Extract appropriate subsequences
            X_sub_seqs = X_seqs[:, :self.sample_len, :]
            
            # Flatten each subsequence to feed into encoder
            batch_size = X_sub_seqs.shape[0]
            X_flat = X_sub_seqs.reshape(batch_size, -1).to(self.device)
            
            # Extract latent means (mu)
            mu, _ = self.encode(X_flat)
            
            # Return as numpy array
            return mu.cpu().numpy()
    
    def get_dynamic_latent_space(self, data_list=None, return_labels=True):
        """
        Create a visualization of the VAE's latent dynamics by encoding overlapping windows
        to see how the latent representation changes as a sequence progresses
        
        Parameters:
        -----------
        data_list : list, optional
            List of data sequences to encode. If None, uses training data.
        return_labels : bool, optional
            Whether to return action labels for each trajectory.
            
        Returns:
        --------
        trajectories : list of numpy.ndarray
            Each element is a sequence's trajectory through latent space
        labels : list, optional
            Class/action labels for each trajectory (if return_labels=True)
        """
        self.eval()
        with torch.no_grad():
            if data_list is None:
                # Use training data
                data_list = self.data_set_class.Y_train_list
                action_labels = self.data_set_class.action_IDs_train
            else:
                # Use provided data (assuming it's test data)
                action_labels = self.data_set_class.action_IDs_test
                
            trajectories = []
            trajectory_labels = []
            
            # Window size for sliding window (how many frames to encode at once)
            window_size = self.sample_len
            
            for i, sequence in enumerate(data_list):
                # Skip sequences that are too short
                if len(sequence) < window_size:
                    continue
                
                # For each position, encode a window of frames
                latent_points = []
                for t in range(len(sequence) - window_size + 1):
                    window = sequence[t:t+window_size]
                    
                    # Convert to tensor and add batch dimension
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Flatten
                    window_flat = window_tensor.reshape(1, -1)
                    
                    # Encode to get latent representation
                    mu, _ = self.encode(window_flat)
                    
                    # Store latent point
                    latent_points.append(mu.cpu().numpy()[0])
                
                # Only include if we have latent points
                if latent_points:
                    # Store trajectory for this sequence
                    trajectories.append(np.array(latent_points))
                    trajectory_labels.append(action_labels[i])
            
            if return_labels:
                return trajectories, trajectory_labels
            else:
                return trajectories

    @staticmethod
    def create_array_with_ones(shape, points):
        arr = np.zeros(shape)
        arr[points] = 1
        return arr

    def IF_setup(self):
        return self.IF_general(self.results_dict['Y_pred_CCs_test'], self.data_set_class.Y_test_CCs)

    def IF_general(self, prediction, ground_truth):
        IFs = []
        for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
            IFs.append(self.data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
        return IFs


class TransformerMotionModel(nn.Module, Evaluation_Base, Architecture_Base):
    def __init__(self, wrapper, data_set_class, sample_len,input_size, num_classes, hidden_size_multiplier=1, num_layers=2,
                 num_heads=4, dropout=0.1, max_seq_length=100):
        super(TransformerMotionModel, self).__init__()
        Evaluation_Base.__init__(self)
        Architecture_Base.__init__(self)
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
    
            self.store_data(epoch, score_rate, num_epochs, loss.item(), self.data_set_class.Y_validation_list, self.data_set_class.Y_validation_CCs, self.data_set_class.action_IDs_validation,'validation')
            self.store_data(epoch, score_rate, num_epochs, loss.item(), self.data_set_class.Y_test_list, self.data_set_class.Y_test_CCs, self.data_set_class.action_IDs_test,'test')
        return self.score(self.data_set_class.Y_test_list, self.data_set_class.Y_test_CCs, 'test', num_epochs, loss.item(), self.data_set_class.action_IDs_test,)


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

    def get_latent_space(self, data_list=None):
        """
        Extract latent space representations from the Transformer model
        
        Parameters:
        -----------
        data_list : list, optional
            List of data sequences to encode. If None, uses training data.
            
        Returns:
        --------
        latent_space : numpy.ndarray
            Matrix of latent representations (N x hidden_size)
        """
        self.eval()
        with torch.no_grad():
            if data_list is None:
                # Use training data by default
                X_seqs = torch.tensor(np.array(self.data_set_class.Y_train_list), dtype=torch.float32)
            else:
                X_seqs = torch.tensor(np.array(data_list), dtype=torch.float32)
            
            # Extract appropriate subsequences
            X_sub_seqs = X_seqs[:, :self.sample_len, :].to(self.device)
            
            # Create attention mask
            attention_mask = (X_sub_seqs != 0).any(dim=-1).float().to(self.device)
            
            # Get hidden states from the transformer
            batch_size = X_sub_seqs.size(0)
            x = self.input_projection(X_sub_seqs)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            if attention_mask is not None:
                cls_attention = torch.ones((batch_size, 1), device=self.device)
                attention_mask = torch.cat((cls_attention, attention_mask), dim=1)

            seq_len = x.size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)

            transformer_output = self.transformer(inputs_embeds=x, attention_mask=attention_mask, position_ids=position_ids)
            
            # Use the [CLS] token representations as latent space
            latent_representations = transformer_output.last_hidden_state[:, 0, :]
            
            # Return as numpy array
            return latent_representations.cpu().numpy()

    def get_dynamic_latent_space(self, data_list=None, return_labels=True):
        """
        Create a visualization of the Transformer's latent dynamics by encoding
        sliding windows of the sequence and extracting the [CLS] token representation
        
        Parameters:
        -----------
        data_list : list, optional
            List of data sequences to encode. If None, uses training data.
        return_labels : bool, optional
            Whether to return action labels for each trajectory.
            
        Returns:
        --------
        trajectories : list of numpy.ndarray
            Each element is a sequence's trajectory through latent space
        labels : list, optional
            Class/action labels for each trajectory (if return_labels=True)
        """
        self.eval()
        with torch.no_grad():
            if data_list is None:
                # Use training data
                data_list = self.data_set_class.Y_train_list
                action_labels = self.data_set_class.action_IDs_train
            else:
                # Use provided data (assuming it's test data)
                action_labels = self.data_set_class.action_IDs_test
                
            trajectories = []
            trajectory_labels = []
            
            # Window size for sliding window (how many frames to encode at once)
            window_size = self.sample_len
            
            for i, sequence in enumerate(data_list):
                # Skip sequences that are too short
                if len(sequence) < window_size:
                    continue
                    
                # For each position, encode a window of frames
                latent_points = []
                for t in range(len(sequence) - window_size + 1):
                    window = sequence[t:t+window_size]
                    
                    # Convert to tensor and add batch dimension
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Create attention mask
                    attention_mask = (window_tensor != 0).any(dim=-1).float().to(self.device)
                    
                    # Get transformer output
                    x = self.input_projection(window_tensor)
                    cls_tokens = self.cls_token.expand(1, -1, -1)
                    x = torch.cat((cls_tokens, x), dim=1)
                    
                    cls_attention = torch.ones((1, 1), device=self.device)
                    full_attention_mask = torch.cat((cls_attention, attention_mask), dim=1)
                    
                    seq_len = x.size(1)
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
                    
                    transformer_output = self.transformer(
                        inputs_embeds=x, 
                        attention_mask=full_attention_mask, 
                        position_ids=position_ids
                    )
                    
                    # Extract [CLS] token representation
                    cls_representation = transformer_output.last_hidden_state[:, 0, :]
                    
                    # Store latent point
                    latent_points.append(cls_representation.cpu().numpy()[0])
                
                # Only include if we have latent points
                if latent_points:
                    # Store trajectory for this sequence
                    trajectories.append(np.array(latent_points))
                    trajectory_labels.append(action_labels[i])
                
            if return_labels:
                return trajectories, trajectory_labels
            else:
                return trajectories


class SequenceClassifier(nn.Module, Evaluation_Base, Architecture_Base):
    def __init__(self, wrapper, data_set_class, sample_len,input_size, num_classes, hidden_size=50, num_layers=2,
                 seq_len=100, **kwargs):
        super(SequenceClassifier, self).__init__()
        Evaluation_Base.__init__(self)
        Architecture_Base.__init__(self)
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

            self.store_data(epoch, score_rate, num_epochs, loss.item(), self.data_set_class.Y_validation_list, self.data_set_class.Y_validation_CCs, self.data_set_class.action_IDs_validation,'validation')
            self.store_data(epoch, score_rate, num_epochs, loss.item(), self.data_set_class.Y_test_list, self.data_set_class.Y_test_CCs, self.data_set_class.action_IDs_test,'test')
        return self.score(self.data_set_class.Y_test_list, self.data_set_class.Y_test_CCs, 'test', num_epochs, loss.item(), self.data_set_class.action_IDs_test,)

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

    def get_latent_space(self, data_list=None):
        """
        Extract latent space representations from the LSTM model
        
        Parameters:
        -----------
        data_list : list, optional
            List of data sequences to encode. If None, uses training data.
            
        Returns:
        --------
        latent_space : numpy.ndarray
            Matrix of latent representations (N x hidden_size)
        """
        self.eval()
        with torch.no_grad():
            if data_list is None:
                # Use training data by default
                num_seqs = len(self.data_set_class.Y_train_list)
                max_seq_len = max(len(seq) for seq in self.data_set_class.Y_train_list)
                X_seqs = np.zeros((num_seqs, max_seq_len, self.data_set_class.Y_train.shape[1]))
                for i, Y_train in enumerate(self.data_set_class.Y_train_list):
                    X_seqs[i, :len(Y_train), :] = Y_train
            else:
                num_seqs = len(data_list)
                max_seq_len = max(len(seq) for seq in data_list)
                X_seqs = np.zeros((num_seqs, max_seq_len, data_list[0].shape[1]))
                for i, seq in enumerate(data_list):
                    X_seqs[i, :len(seq), :] = seq
            
            # Convert to tensor and move to device
            X_seqs_tensor = torch.from_numpy(X_seqs).float().to(self.device)
            
            # Extract appropriate subsequences
            X_sub_seqs = X_seqs_tensor[:, :self.sample_len, :]
            
            # Pass through LSTM and get hidden state
            lstm_out, (hn, cn) = self.lstm(X_sub_seqs)
            
            # Use the final hidden state as latent representation
            # Take the last layer's hidden state for each sequence
            latent_representations = hn[-1, :, :]
            
            # Return as numpy array
            return latent_representations.cpu().numpy()

    def get_dynamic_latent_space(self, data_list=None, return_labels=True):
        """
        Extract hidden state trajectories from the LSTM model for each sequence
        
        Parameters:
        -----------
        data_list : list, optional
            List of data sequences to encode. If None, uses training data.
        return_labels : bool, optional
            Whether to return action labels for each trajectory.
            
        Returns:
        --------
        trajectories : list of numpy.ndarray
            Each element is a sequence's trajectory through hidden space
        labels : list, optional
            Class/action labels for each trajectory (if return_labels=True)
        """
        self.eval()
        with torch.no_grad():
            if data_list is None:
                # Use training data
                data_list = self.data_set_class.Y_train_list
                action_labels = self.data_set_class.action_IDs_train
            else:
                # Use provided data (assuming it's test data)
                action_labels = self.data_set_class.action_IDs_test
            
            trajectories = []
            trajectory_labels = []
            
            for i, sequence in enumerate(data_list):
                # Skip if sequence is too short
                if len(sequence) < 2:  # Need at least 2 time points to show a trajectory
                    continue
                    
                # Convert to tensor and add batch dimension
                seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Initialize hidden and cell states
                batch_size = 1
                h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
                c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
                
                # Process each timestep and collect hidden states
                hidden_states = []
                for t in range(seq_tensor.size(1)):
                    # Get single timestep
                    x_t = seq_tensor[:, t:t+1, :]
                    
                    # Process through LSTM
                    _, (h_t, c_t) = self.lstm(x_t, (h0, c0))
                    
                    # Store hidden state (using last layer)
                    hidden_states.append(h_t[-1, 0].cpu().numpy())
                    
                    # Update hidden state for next timestep
                    h0, c0 = h_t, c_t
                
                # Only include if we have hidden states
                if hidden_states:
                    # Store trajectory for this sequence
                    trajectories.append(np.array(hidden_states))
                    trajectory_labels.append(action_labels[i])
            
            if return_labels:
                return trajectories, trajectory_labels
            else:
                return trajectories