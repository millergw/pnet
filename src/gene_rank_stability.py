"""
Copied from jupyter notebook so that it can be executed as a script.
In this script, we analyse the stability of the layer conductance values from the model on a fixed train and test set. We are interested in how the random initialization of the weights influences the importance scores obtained for the genes.

"""
import json
import Pnet
import report_and_eval
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import wandb
import pickle
import logging
logging.basicConfig(
            filename='run_pnet.log', 
            encoding='utf-8',
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

wandb.login()

logging.info("Setting data and save directories")
# DATADIR = '/mnt/disks/pancan' # Marc's
DATADIR = '../../pnet_germline/data' # Gwen's
SAVEDIR = '../../pnet/results/gene_rank_stability' # Gwen's
report_and_eval.make_dir_if_needed(SAVEDIR)

logging.info("Loading data")
prostate_mutations = pd.read_csv(os.path.join(DATADIR, 'pnet_database/prostate/processed/P1000_final_analysis_set_cross_important_only.csv'))
prostate_mutations.set_index('Tumor_Sample_Barcode', inplace=True)

prostate_cnv = pd.read_csv(os.path.join(DATADIR, 'pnet_database/prostate/processed/P1000_data_CNA_paper.csv'))
prostate_cnv.rename(columns={"Unnamed: 0": "Tumor_Sample_Barcode"}, inplace=True)
prostate_cnv.set_index('Tumor_Sample_Barcode', inplace=True)

prostate_response = pd.read_csv(os.path.join(DATADIR, 'pnet_database/prostate/processed/response_paper.csv'))
prostate_response.rename(columns={'id': "Tumor_Sample_Barcode"}, inplace=True)
prostate_response.set_index('Tumor_Sample_Barcode', inplace=True)

prostate_genes = pd.read_csv(os.path.join(DATADIR, 'pnet_database/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv'))
prostate_genes = list(set(prostate_genes['genes']).intersection(set(prostate_mutations.columns)).intersection(set(prostate_cnv.columns)))

# prostate_mutations = pd.read_csv('../../data/pnet_database/prostate/processed/P1000_final_analysis_set_cross_important_only.csv')
# prostate_mutations.set_index('Tumor_Sample_Barcode', inplace=True)

# prostate_cnv = pd.read_csv('../../data/pnet_database/prostate/processed/P1000_data_CNA_paper.csv')
# prostate_cnv.rename(columns={"Unnamed: 0": "Tumor_Sample_Barcode"}, inplace=True)
# prostate_cnv.set_index('Tumor_Sample_Barcode', inplace=True)

# prostate_response = pd.read_csv('../../data/pnet_database/prostate/processed/response_paper.csv')
# prostate_response.rename(columns={'id': "Tumor_Sample_Barcode"}, inplace=True)
# prostate_response.set_index('Tumor_Sample_Barcode', inplace=True)

prostate_genes = pd.read_csv(os.path.join(DATADIR, 'pnet_database/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv'))
prostate_genes = list(set(prostate_genes['genes']).intersection(set(prostate_mutations.columns)).intersection(set(prostate_cnv.columns)))

prostate_cnv = prostate_cnv[prostate_genes].copy()
prostate_mutations = prostate_mutations[prostate_genes].copy()

# prostate_genes = util.select_highly_variable_genes(prostate_mutations)
# prostate_genes = prostate_genes['level_1']
prostate_mutations = prostate_mutations[list(set(prostate_mutations.columns).intersection(prostate_genes))].copy()
prostate_cnv = prostate_cnv[list(set(prostate_cnv.columns).intersection(prostate_genes))].copy()

# Regenerate input as specified in prostate_paper
prostate_mutations = (prostate_mutations > 0).astype(int)
prostate_amp = (prostate_cnv > 1).astype(int)
prostate_del = (prostate_cnv < -1).astype(int)

genetic_data = {'mut': prostate_mutations, 
                'amp': prostate_amp, 
                'del': prostate_del}
logging.info(f"We are using the datasets: {list(genetic_data.keys())}")

logging.info("Defining train/test indices")
test_inds = list(pd.read_csv(os.path.join(DATADIR, 'pnet_database/prostate/splits/test_set.csv'))['id'])
train_inds = list(pd.read_csv(os.path.join(DATADIR, 'pnet_database/prostate/splits/training_set.csv'))['id'])

# test_inds = list(pd.read_csv('/mnt/disks/pancan/pnet_database/splits/test_set.csv')['id'])
# train_inds = list(pd.read_csv('/mnt/disks/pancan/pnet_database/splits/training_set.csv')['id'])


logging.info("Model training")
gene_imps = []
layerwise_imps = []
aucs = []
for r in range(20):
    wandb.init(
        # Set the project where this run will be logged
        project="prostate_met_status",
        name=f"gene_stability_somatic_{r}"
    )
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data,
                                                                         prostate_response,
                                                                         verbose=False,
                                                                         early_stopping=False,
                                                                         train_inds=train_inds,
                                                                         test_inds=test_inds)
    model.to('cpu')
    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y
    pred = model(x_test, additional_test)
    y_pred_proba = pred[0].detach().numpy().squeeze()
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    aucs.append(metrics.roc_auc_score(y_test, y_pred_proba))
    gene_imps.append(model.gene_importance(test_dataset))
    layerwise_imps.append(model.layerwise_importance(test_dataset))

    # logging.info(f"Updating gene importances, layer importances, and AUCs to files in {SAVEDIR}")
    # # Save each list by appending to the file in each iteration
    # for lst, filename in zip([gene_imps, layerwise_imps, aucs], 
    #                          ['gene_imps.json', 'layerwise_imps.json', 'aucs.json']):
    #     # Open the file in append mode
    #     with open(os.path.join(SAVEDIR, filename), 'a') as f:
    #         # Save the updated list to the file
    #         json.dump(lst, f)
            
    #         # Add a newline for readability
    #         f.write('\n')
    # with open(os.path.join(SAVEDIR, 'gene_imps.json'), 'a') as f:
    #     json.dump(gene_imps, f)

    # with open(os.path.join(SAVEDIR, 'layerwise_imps.json'), 'a') as f:
    #     json.dump(layerwise_imps, f)

    # with open(os.path.join(SAVEDIR, 'aucs.json'), 'a') as f:
    #     json.dump(aucs, f)
    logging.debug(f"gene_imp: \n{model.gene_importance(test_dataset)}\nlayerwise_imp: \n{model.layerwise_importance(test_dataset)}\nauc: \n{metrics.roc_auc_score(y_test, y_pred_proba)}")

    logging.info(f"convergence for model {r}")
    plt = report_and_eval.get_loss_plot(train_losses=train_scores, test_losses=test_scores)
    report_and_eval.savefig(plt, os.path.join(SAVEDIR, f'loss_over_time_{r}'))
    plt.show()
    
    wandb.finish()
    
logging.info(f"Saving gene importances, layer importances, and AUCs to files in {SAVEDIR}")
# with open(os.path.join(SAVEDIR, 'gene_imps.json'), 'w') as f:
#     json.dump(gene_imps, f)

# with open(os.path.join(SAVEDIR, 'layerwise_imps.json'), 'w') as f:
#     json.dump(layerwise_imps, f)

# with open(os.path.join(SAVEDIR, 'aucs.json'), 'w') as f:
#     json.dump(aucs, f)

# Save gene_imps to a Pickle file
with open(os.path.join(SAVEDIR, 'gene_imps.pkl'), 'wb') as file:
    pickle.dump(gene_imps, file)

# Save layerwise_imps to a Pickle file
with open(os.path.join(SAVEDIR, 'layerwise_imps.pkl'), 'wb') as file:
    pickle.dump(layerwise_imps, file)

# Save aucs to a Pickle file
with open(os.path.join(SAVEDIR, 'aucs.pkl'), 'wb') as file:
    pickle.dump(aucs, file)

logging.info("Initial exploratory stuff")
pd.concat(gene_imps, axis=1).std(axis=1).nlargest(20)
pd.concat([gis.rank(ascending=False) for gis in gene_imps], axis=1).mean(axis=1).nsmallest(20)