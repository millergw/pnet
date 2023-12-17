import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAUROC
import numpy as np
import os
import warnings
import pytorch_lightning as pl
import captum
import ReactomeNetwork
import GenesetNetwork
import pnet_loader
from CustomizedLinear import masked_activation
import util
import wandb
import random


class PNET_Block(nn.Module):
    def __init__(self, gene_mask, pathway_mask, activation='tanh', batchnorm=False, gene_dropout=None, dropout=None):
        """
        Pathway level unit of deep network. Taking in connections from the gene level and the previous pathway level.
        Combines the two inputs by addition, applies a batchnorm, non-linearity and dropout before passing it to the
        higher order pathway level.
        :param gene_mask: np.array; binary adjacency matrix from gene level to pathways in layer
        :param pathway_mask: np.array; binary adjacency matrix from previous layer to pathways in current layer
        :param dropout: float; fraction of connections to randomly drop out, applied on layer output
        """
        super(PNET_Block, self).__init__()
        self.gene_layer = nn.Sequential(*masked_activation(gene_mask, activation=activation, batchnorm=batchnorm, dropout=gene_dropout))
        self.pathway_layer = nn.Sequential(*masked_activation(pathway_mask, activation=activation, batchnorm=batchnorm, dropout=dropout))

    def forward(self, x, genes):
        x_genes = self.gene_layer(genes)
        x_pathway = self.pathway_layer(x)
        return x_genes + x_pathway
    
    
class Regulatory_Block(nn.Module):
    def __init__(self, regulatory_mask, activation='tanh', batchnorm=False, gene_dropout=None, dropout=None):
        """
        Regulatory level unit of deep network. Taking in connections from the gene level and the previous pathway level.
        Combines the two inputs by addition, applies a batchnorm, non-linearity and dropout before passing it to the
        gene layer level.
        :param regulatory_mask: np.array; binary adjacency matrix from gene level to pathways in layer
        :param dropout: float; fraction of connections to randomly drop out, applied on layer output
        """
        super(Regulatory_Block, self).__init__()
        self.regulatory_layer = nn.Sequential(*masked_activation(regulatory_mask, activation=activation, batchnorm=batchnorm, dropout=gene_dropout))

    def forward(self, x):
        x_regulatory = self.regulatory_layer(x)
        return x_regulatory + x


class PNET_NN(pl.LightningModule):
    def __init__(self, reactome_network, task, nbr_gene_inputs=1, output_dim=1, additional_dims=0, lr=1e-3, weight_decay=1e-5,
                 dropout=0.1, gene_dropout=0.1, input_dropout=0.5, activation='tanh', loss_fn=None, random_network=False, fcnn=False,
                 loss_weight=None, aux_loss_weights=[2, 7, 20, 54, 148, 400], add_regulatory_layer=False):
        super().__init__()
        self.reactome_network = reactome_network
        self.nbr_gene_inputs = nbr_gene_inputs
        self.output_dim = output_dim
        self.additional_dims =additional_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.gene_dropout = gene_dropout
        self.input_dropout = input_dropout
        self.task = task
        self.loss_weight = loss_weight
        self.aux_loss_weights = aux_loss_weights
        if loss_fn is None:
            self.loss_fn = util.get_loss_function(task)
        else:
            self.loss_fn = loss_fn
        self.activation = activation
        self.interpret_flag = False
        self.regulatory_flag = add_regulatory_layer
        
        # Fetch connection masks from reactome network:
        if self.regulatory_flag:
            gene_masks, pathway_masks, input_mask, regulatory_mask = self.reactome_network.get_masks(self.nbr_gene_inputs, regulatory=True)
        else:
            gene_masks, pathway_masks, input_mask = self.reactome_network.get_masks(self.nbr_gene_inputs)
        if random_network:
            for gm in gene_masks:
                util.shuffle_connections(gm)
            for pm in pathway_masks:
                util.shuffle_connections(pm)
                
        if fcnn:
            gene_masks = [np.ones_like(gm) for gm in gene_masks]
            pathway_masks = [np.ones_like(gm) for gm in pathway_masks]
        
        self.num_pred_heads = len(gene_masks)
        
        # Prepare list of layers and list of predictions per layer:
        self.layers = nn.ModuleList()
        self.preds = nn.ModuleList()
        # Add input layer to aggregate all data modalities
        self.input_layer = nn.Sequential(*masked_activation(input_mask, activation=self.activation, batchnorm=True, dropout=self.input_dropout))
        # Add regulatory layer if active
        if self.regulatory_flag:   
            self.regulatory_layer = Regulatory_Block(regulatory_mask, activation=self.activation, batchnorm=True, dropout=self.dropout)
            self.regulatory_pred = nn.Sequential(*[nn.Linear(in_features=regulatory_mask.shape[0] + self.additional_dims, out_features=self.output_dim)])
            self.num_pred_heads += 1
        # Add first layer separately:
        self.first_gene_layer = nn.Sequential(*masked_activation(gene_masks[0], activation=self.activation, batchnorm=True, dropout=self.gene_dropout))
        # Add blocks and prediction heads for each pathway level:
        for i in range(0, len(gene_masks) - 1):
            self.layers.append(PNET_Block(gene_masks[i + 1], pathway_masks[i], activation=self.activation, batchnorm=True, dropout=self.dropout))
            self.preds.append(
                nn.Sequential(*[nn.Linear(in_features=pathway_masks[i].shape[0] + self.additional_dims,
                                          out_features=self.output_dim)]))
        # Add final prediction layer:
        self.preds.append(nn.Sequential(*[nn.Linear(in_features=pathway_masks[len(gene_masks) - 1].shape[0] +
                                                                self.additional_dims, out_features=self.output_dim)]))
        # Weighting of the different prediction layers:
        self.attn = nn.Linear(in_features=(self.num_pred_heads) * self.output_dim, out_features=self.output_dim)

    def forward(self, x, additional_data):
        x = self.input_layer(x)
        y_hats = []
        
        # Apply regulatory layer and predict from it
        if self.regulatory_flag:
            x = self.regulatory_layer(x)
            x_cat = torch.concat([x, additional_data], dim=1)
            y_hats.append(self.regulatory_pred(x_cat))
        
        # Fix gene layer for downstream inputs
        genes = torch.clone(x)
        
        # Apply first gene to pathway layer and predict from it
        x = self.first_gene_layer(x)
        x_cat = torch.concat([x, additional_data], dim=1)
        y_hats.append(self.preds[0](x_cat))
        
        # Iterate through all other pathway layers
        for layer, pred in zip(self.layers, self.preds[1:]):
            x = layer(x, genes)
            x_cat = torch.concat([x, additional_data], dim=1)
            y_hats.append(pred(x_cat))
            
        # Generate final prediction by weighting all predictions
        y = self.attn(torch.concat(y_hats, dim=1))
        
        # return only final prediction if in interpret mode or all predictions of all levels if in training
        if self.interpret_flag:
            return y
        else:
            return y, y_hats

    def step(self, who, batch, batch_nb):
        x, additional, y = batch
        pred_y, _ = self(x, additional)
        loss = F.cross_entropy(pred_y, y, reduction='mean')

        self.log(who + '_bce_loss', loss)
        return loss
    
    def predict_proba(self,  x, additional_data, threshold=0.5):
        logits, lower_level_logits = self.forward(x, additional_data)
        if self.task == 'BC':
            probabilities = torch.sigmoid(logits)
            return probabilities
        elif self.task == 'MC':
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(logits)
            return probabilities
        else:
            raise Exception("Trying to calculate class probabilies with a regression model")
            return logits

    def predict(self,  x, additional_data, threshold=0.5):
        logits, lower_level_logits = self.forward(x, additional_data)
        if self.task == 'BC':
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            return predictions
        elif self.task == 'MC':
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(logits)
            _, predictions = probabilities.max(dim=1)
            binary_predictions = (probabilities > threshold).float()
            return binary_predictions
        else:
            return logits


    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def deepLIFT(self, test_dataset, target_class=0):
        self.interpret_flag=True
        dl = captum.attr.DeepLift(self)
        gene_importances, additional_importances = dl.attribute((test_dataset.x, test_dataset.additional)
                                                                , target=target_class)
        gene_importances = pd.DataFrame(gene_importances.detach().numpy(),
                                        index=test_dataset.input_df.index,
                                        columns=test_dataset.input_df.columns)
        additional_importances = pd.DataFrame(additional_importances.detach().numpy(),
                                              index=test_dataset.additional_data.index,
                                              columns=test_dataset.additional_data.columns)
        self.gene_importances, self.additional_importances = gene_importances, additional_importances
        self.interpret_flag=False
        return self.gene_importances, self.additional_importances
    
    def integrated_gradients(self, test_dataset, target_class=0):
        self.interpret_flag=True
        ig = captum.attr.IntegratedGradients(self)
        if self.task == 'REG':
            ig_attr = ig.attribute((test_dataset.x, test_dataset.additional), n_steps=50)
        else:
            ig_attr, delta = ig.attribute((test_dataset.x, test_dataset.additional), return_convergence_delta=True, target=target_class)
        gene_importances, additional_importances = ig_attr
        gene_importances = pd.DataFrame(gene_importances.detach().numpy(),
                                        index=test_dataset.input_df.index,
                                        columns=test_dataset.input_df.columns)
        additional_importances = pd.DataFrame(additional_importances.detach().numpy(),
                                              index=test_dataset.additional_data.index,
                                              columns=test_dataset.additional_data.columns)
        self.gene_importances, self.additional_importances = gene_importances, additional_importances
        self.interpret_flag=False
        return self.gene_importances, self.additional_importances

    def layerwise_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        cond = captum.attr.LayerConductance(self, self.first_gene_layer)  # ReLU output of masked layer at each level
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[0].index]
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        pathway_imp_by_target = cond_vals_genomic
        layer_importance_scores.append(pathway_imp_by_target)
        
        for i, level in enumerate(self.layers):
            cond = captum.attr.LayerConductance(self, level.pathway_layer)  # ReLU output of masked layer at each level
            cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
            cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[i].columns]
            cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                             columns=cols,
                                             index=test_dataset.input_df.index)
            pathway_imp_by_target = cond_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
        self.interpret_flag=False
        return layer_importance_scores
    
    def layerwise_activation(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        for i, level in enumerate(self.layers):
            act = captum.attr.LayerActivation(self, level.pathway_layer)
            act_vals = act.attribute((test_dataset.x, test_dataset.additional), attribute_to_layer_input=True)
            cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[i].index]
            act_vals_genomic = pd.DataFrame(act_vals.detach().numpy(),
                                            columns=cols,
                                            index=test_dataset.input_df.index)
            pathway_imp_by_target = act_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
        self.interpret_flag=False
        return layer_importance_scores
    
    def neuron_conductance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        layer_importance_scores = []
        for i, level in enumerate(self.layers):
            neuron_cond = captum.attr.NeuronConductance(self, level.pathway_layer)
            neuron_cond_att = neuron_cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
            
        self.interpret_flag=False    
    
    def gene_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        cond = captum.attr.LayerConductance(self, self.input_layer)
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = self.reactome_network.gene_list
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        gene_imp_by_target = cond_vals_genomic
        self.interpret_flag=False
        return gene_imp_by_target
    
    def regulatory_layer_importance(self, test_dataset, target_class=0):
        self.interpret_flag=True
        cond = captum.attr.LayerConductance(self, self.regulatory_layer.regulatory_layer)
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = self.reactome_network.gene_list
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        gene_imp_by_target = cond_vals_genomic
        self.interpret_flag=False
        return gene_imp_by_target
    
    def interpret(self, test_dataset, plot=False):
        gene_feature_importances, additional_feature_importances = self.integrated_gradients(test_dataset)
        gene_importances = self.gene_importance(test_dataset)
        # layer_importance_scores = self.layerwise_importance(test_dataset)
        if self.regulatory_flag == True:
            regulatory_importances = self.regulatory_layer_importance(test_dataset)

        layer_importance_scores = self.layerwise_importance(test_dataset)
        
        gene_order = gene_importances.mean().sort_values(ascending=True).index
        if plot:
            plt.rcParams["figure.figsize"] = (6,8)
            gene_importances[list(gene_order[-20:])].plot(kind='box', vert=False)
            plt.savefig(plot+'/imp_genes.pdf')
        self.interpret_flag=False
        if self.regulatory_flag == True:
            return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores, regulatory_importances
        else: 
            return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores,

    def interpret(self, test_dataset, plot=False):
        gene_feature_importances, additional_feature_importances = self.integrated_gradients(test_dataset)
        gene_importances = self.gene_importance(test_dataset)
        # layer_importance_scores = self.layerwise_importance(test_dataset)
        layer_importance_scores = self.layerwise_importance(test_dataset)
        
        gene_order = gene_importances.mean().sort_values(ascending=True).index
        if plot:
            plt.rcParams["figure.figsize"] = (6,8)
            gene_importances[list(gene_order[-20:])].plot(kind='box', vert=False)
            plt.savefig(plot+'/imp_genes.pdf')
        self.interpret_flag=False
        return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores


def fit(model, dataloader, optimizer):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        gene_data, additional_data, y = batch
        gene_data, additional_data, y = gene_data.to(device), additional_data.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat, y_hats = model(gene_data, additional_data)
        if model.loss_weight is not None:
            weight = model.loss_weight.to(device)
            weight_ = weight[y.data.view(-1).long()].view_as(y)
            aux_losses = [(model.loss_fn(y_h, y) * weight_).mean() * w for y_h, w in zip(y_hats, model.aux_loss_weights)]
            loss = (model.loss_fn(y_hat, y) * weight_).mean() + sum(aux_losses)
        else:
            aux_losses = [model.loss_fn(y_h, y) * w for y_h, w in zip(y_hats, model.aux_loss_weights)]
            loss = model.loss_fn(y_hat, y) + sum(aux_losses)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def validate(model, dataloader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.eval()
    running_loss = 0.0
    for batch in dataloader:
        gene_data, additional_data, y = batch
        gene_data, additional_data, y = gene_data.to(device), additional_data.to(device), y.to(device)
        y_hat, y_hats = model(gene_data, additional_data)
        if model.loss_weight is not None:
            weight = model.loss_weight.to(device)
            weight_ = weight[y.data.view(-1).long()].view_as(y)
            aux_losses = [(model.loss_fn(y_h, y) * weight_).mean() * w for y_h, w in zip(y_hats, model.aux_loss_weights)]
            loss = (model.loss_fn(y_hat, y) * weight_).mean() + sum(aux_losses)
        else:
            aux_losses = [model.loss_fn(y_h, y) * w for y_h, w in zip(y_hats, model.aux_loss_weights)]
            loss = model.loss_fn(y_hat, y) + sum(aux_losses)
        running_loss += loss.item()
        loss.backward()
    loss = running_loss / len(dataloader.dataset)
    return loss


def train(model, train_loader, test_loader, save_path, lr=0.5e-3, weight_decay=1e-4, epochs=300, verbose=False,
          early_stopping=True, lr_scheduler=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We are sending to cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=lr)
    early_stopper = util.EarlyStopper(save_path, patience=50, min_delta=0.01, verbose=verbose)
    train_scores = []
    test_scores = []
    for epoch in range(epochs):
        train_epoch_loss = fit(model, train_loader, optimizer)
        test_epoch_loss = validate(model, test_loader)
        train_scores.append(train_epoch_loss)
        test_scores.append(test_epoch_loss)
        # saving info to Weights and Biases
        wandb.log({
            "Train Loss": train_epoch_loss,
            "Test Loss": test_epoch_loss
        })

        if lr_scheduler:
            scheduler.step()
        if verbose:
            print(f"Epoch {epoch + 1} of {epochs}")
            print("Train Loss: {}".format(train_epoch_loss))
            print("Test Loss: {}".format(test_epoch_loss))
        if early_stopper.early_stop(test_epoch_loss, model) and early_stopping:
            print('Hit early stopping criteria')
            model.load_state_dict(torch.load(save_path))
            break
    return model, train_scores, test_scores


def evaluate_interpret_save(model, test_dataset, path):
    if not os.path.exists(path):
        os.makedirs(path)
    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y
    model.to('cpu')
    pred_proba = model.predict_proba(x_test, additional_test).detach()
    pred = model.predict(x_test, additional_test).detach()
    auc_score = util.get_auc(pred_proba, y_test, save=path+'/auc_curve.pdf')
    auc_prc = util.get_auc_prc(pred_proba, y_test)
    f1_score = util.get_f1(pred, y_test)
    
    torch.save(pred_proba, path+'/prediction_probabilities.pt')
    torch.save(auc_score, path+'/AUC.pt')
    torch.save(auc_prc, path+'/AUC_PRC.pt')
    torch.save(f1_score, path+'/F1.pt')
    gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = model.interpret(test_dataset)
    gene_feature_importances.to_csv(path+'/gene_feature_importances.csv')
    additional_feature_importances.to_csv(path+'/additional_feature_importances.csv')
    gene_importances.to_csv(path+'/gene_importances.csv')
    for i, layer in enumerate(layer_importance_scores):
        layer.to_csv(path+'/layer_{}_importances.csv'.format(i))



def run(genetic_data, target, save_path='../results/model', gene_set=None, additional_data=None, test_split=0.2, seed=None, dropout=0.2,
        input_dropout=0.5, lr=1e-3, weight_decay=1e-3, batch_size=64, epochs=400, verbose=False, early_stopping=True, train_inds=None,
        test_inds=None, random_network=False, fcnn=False, task=None, loss_fn=None, loss_weight=None, aux_loss_weights=[2, 7, 20, 54, 148, 400]):
    if task is None:
        task = util.get_task(target)
    target = util.format_target(target, task)
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target, gene_set, additional_data,
                                                                  test_split, seed, train_inds, test_inds)
    
    reactome_network = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())

    model = PNET_NN(reactome_network=reactome_network, task=task, nbr_gene_inputs=len(genetic_data), dropout=dropout,
                    additional_dims=train_dataset.additional_data.shape[1], lr=lr, weight_decay=weight_decay,
                    output_dim=target.shape[1], random_network=random_network, fcnn=fcnn, loss_fn=loss_fn, loss_weight=loss_weight,
                    input_dropout=input_dropout, aux_loss_weights=aux_loss_weights
                    )
    train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, batch_size)
    model, train_scores, test_scores = train(model, train_loader, test_loader, save_path, lr, weight_decay, epochs, verbose,
                                             early_stopping)

    return model, train_scores, test_scores, train_dataset, test_dataset


def run_regulatory(genetic_data, target, save_path='../results/model', gene_set=None, additional_data=None, test_split=0.2, seed=None,
                   dropout=0.2, input_dropout=0.5, lr=1e-3, weight_decay=1e-3, batch_size=64, epochs=400, verbose=False, early_stopping=True,
                   train_inds=None, test_inds=None, random_network=False, fcnn=False, task=None, loss_fn=None, loss_weight=None,
                   aux_loss_weights=[25, 2, 7, 20, 54, 148, 400]):
    if task is None:
        task = util.get_task(target)
    target = util.format_target(target, task)
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target, gene_set, additional_data,
                                                                  test_split, seed, train_inds, test_inds)
    
    reactome_network = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())

    model = PNET_NN(reactome_network=reactome_network, task=task, nbr_gene_inputs=len(genetic_data), dropout=dropout,
                    additional_dims=train_dataset.additional_data.shape[1], lr=lr, weight_decay=weight_decay,
                    output_dim=target.shape[1], random_network=random_network, fcnn=fcnn, loss_fn=loss_fn, loss_weight=loss_weight,
                    input_dropout=input_dropout, aux_loss_weights=aux_loss_weights, add_regulatory_layer=True
                    )
    train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, batch_size)
    model, train_scores, test_scores = train(model, train_loader, test_loader, save_path, lr, weight_decay, epochs, verbose, early_stopping)
    return model, train_scores, test_scores, train_dataset, test_dataset


def run_geneset(genetic_data, target, geneset_path, num_layers=3, sparsity=0.9, save_path='../results/model', genes=None, additional_data=None,
                test_split=0.2, seed=None, dropout=0.2, input_dropout=0.5, lr=1e-3, weight_decay=1e-3, batch_size=64, epochs=400, verbose=False,
                early_stopping=True, train_inds=None, test_inds=None, random_network=False, fcnn=False, task=None, loss_fn=None, loss_weight=None,
                aux_loss_weights=[2, 7, 20, 54, 148, 400]):
    if task is None:
        task = util.get_task(target)
    target = util.format_target(target, task)
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target, genes, additional_data,
                                                                  test_split, seed, train_inds, test_inds)
    
    geneset_network = GenesetNetwork.GenesetNetwork(train_dataset.get_genes(), path=geneset_path, num_layers=num_layers, sparsity=sparsity, trim=0)

    model = PNET_NN(reactome_network=geneset_network, task=task, nbr_gene_inputs=len(genetic_data), dropout=dropout,
                    additional_dims=train_dataset.additional_data.shape[1], lr=lr, weight_decay=weight_decay,
                    output_dim=target.shape[1], random_network=random_network, fcnn=fcnn, loss_fn=loss_fn, loss_weight=loss_weight,
                    input_dropout=input_dropout, aux_loss_weights=aux_loss_weights
                    )
    train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, batch_size)
    model, train_scores, test_scores = train(model, train_loader, test_loader, save_path, lr, weight_decay, epochs, verbose,
                                             early_stopping)

    return model, train_scores, test_scores, train_dataset, test_dataset


def interpret(model, x, additional,  plots=False, savedir=''):
    '''
    Function to use DeepLift from Captum on PNET model structure. Generates overall feature importance and layerwise
    results.
    :param model: NN model to predict feature importance on. Assuming PNET structure
    :param data: PnetDataset; data object with samples to use gradients on.
    :return:
    '''
    if plots:
        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
        else:
            savedir = os.getcwd()
    feature_importance = dict()
    # Overall feature importance
    ig = IntegratedGradients(model)
    ig_attr, delta = ig.attribute((x, additional), return_convergence_delta=True)
    ig_attr_genomic, ig_attr_additional = ig_attr
    feature_importance['overall_genomic'] = ig_attr_genomic.detach().numpy()
    feature_importance['overall_clinical'] = ig_attr_additional.detach().numpy()
    if plots:
        visualize_importances(test_df.columns[:clinical_index],
                              np.mean(feature_importance['overall_clinical'], axis=0),
                              title="Average Feature Importances",
                              axis_title="Clinical Features")
        plt.savefig('/'.join([ savedir, 'feature_importance_overall_clinical.pdf']))

        visualize_importances(test_df.columns[clinical_index:],
                              np.mean(feature_importance['overall_genomic'], axis=0),
                              title="Average Feature Importances",
                              axis_title="Genomic Features")
        plt.savefig('/'.join([savedir, 'feature_importance_overall_genomic.pdf']))

    # Neurons feature importance
    layer_importance_scores = []
    for level in model.layers:
        cond = LayerConductance(model, level.activation)       # ReLU output of masked layer at each level
        cond_vals = cond.attribute((genomic_input, clinical_input))
        cond_vals_genomic = cond_vals.detach().numpy()
        layer_importance_scores.append(cond_vals_genomic)
    feature_importance['layerwise_neurons_genomic'] = layer_importance_scores
    if plots:
        for i, layer in enumerate(feature_importance['layerwise_neurons_genomic']):
            pathway_names = model.reactome_network.pathway_encoding.set_index('ID')
            pathway_names = pathway_names.loc[model.reactome_network.pathway_layers[i+1].index]['pathway']
            visualize_importances(pathway_names,
                                  np.mean(layer, axis=0),
                                  title="Neurons Feature Importances",
                                  axis_title="Pathway activation Features")
            plt.savefig('/'.join([savedir, 'pathway_neurons_layer_{}_importance.pdf'.format(i)]))

    return feature_importance


def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, rotation=90)
        plt.xlabel(axis_title)
        plt.title(title)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)