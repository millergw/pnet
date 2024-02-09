"""
Copied from jupyter notebook so that it can be executed as a script.
In this script, we analyse the stability of the layer conductance values from the model on a fixed train and test set. We are interested in how the random initialization of the weights influences the importance scores obtained for the genes.

"""
import Pnet
import pnet_loader
import report_and_eval
import model_selection
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
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

wandb.login()
MODEL_TYPE = "bdt"
EVAL_SET = 'test' # val

logging.info("Setting data and save directories")
# DATADIR = '/mnt/disks/pancan' # Marc's
DATADIR = '../../pnet_germline/data' # Gwen's
SAVEDIR = f'../../pnet/results/somatic_{MODEL_TYPE}_eval_set_{EVAL_SET}' # Gwen's
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

genetic_data = {'somatic_mut': prostate_mutations, 
                'somatic_amp': prostate_amp, 
                'somatic_del': prostate_del,}
logging.info(f"We are using the datasets: {list(genetic_data.keys())}")

logging.info("Defining train/test indices")
test_inds_f = os.path.join(DATADIR, 'pnet_database/prostate/splits/test_set.csv')
test_inds = list(pd.read_csv(test_inds_f)['id'])
train_inds_f = os.path.join(DATADIR, 'pnet_database/prostate/splits/training_set.csv')
train_inds = list(pd.read_csv(train_inds_f)['id'])

logging.info("Loading data and making data splits")
train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target=prostate_response, train_inds=train_inds, test_inds=test_inds, gene_set=prostate_genes)
x_train = train_dataset.x
additional_train = train_dataset.additional
y_train = train_dataset.y.ravel()
x_test = test_dataset.x
additional_test = test_dataset.additional
y_test = test_dataset.y.ravel()

logging.info("Model training")
gene_imps = []
layerwise_imps = []
aucs = []
for r in range(20):
    wandb.init(
        # Set the project where this run will be logged
        project="prostate_met_status",
        name=f"{MODEL_TYPE}_eval_set_{EVAL_SET}_somatic_{r}",
        group=f'{MODEL_TYPE}_stability_experiment_004'
    )
    wandb.config.update(
        {
        'train_inds_f':train_inds_f,
        'test_inds_f':test_inds_f,
        'train_set_indices':train_inds,
        'test_set_indices':test_inds,
        'dataset':list(genetic_data.keys()),
        'model_type':MODEL_TYPE,
        'eval_set':EVAL_SET,
        'save_dir':SAVEDIR,
        'data_dir':DATADIR,
    }
    )

    if MODEL_TYPE == 'pnet':
        model, train_dataset, test_dataset, train_scores, test_scores, auc, gene_imp, layerwise_imp = model_selection.run_pnet(genetic_data, prostate_response, train_inds, test_inds)
        aucs.append(auc)
        gene_imps.append(gene_imp)
        layerwise_imps.append(layerwise_imp)

        # TODO: change from 'pnet_dataset' to x and y?
        logging.info(f"Getting the model predictions, performance metrics, feature importances (as appropriate), and save the results to {SAVEDIR}.")
        report_and_eval.evaluate_interpret_save(model=model, pnet_dataset=train_dataset, model_type=MODEL_TYPE, who="train", save_dir=SAVEDIR)
        report_and_eval.evaluate_interpret_save(model=model, pnet_dataset=test_dataset, model_type=MODEL_TYPE, who=EVAL_SET, save_dir=SAVEDIR) # TODO: who=val/test is hardcoded
        
        logging.info(f"Making loss plots to check convergence for model {r}")
        plt = report_and_eval.get_loss_plot(train_losses=train_scores, test_losses=test_scores)
        report_and_eval.savefig(plt, os.path.join(SAVEDIR, f'loss_over_time_{r}'))
        plt.show()
    elif MODEL_TYPE == 'rf':
        # TODO: 1/10/24. do I need to somehow take the "additional" data into account? Can I just merge this into x_train to create one larger input?
        # x_train, x_test, y_train, y_test = report_and_eval.get_train_test_manual_split(x, y, train_inds, test_inds)
        model = model_selection.run_rf(x_train, y_train, random_seed=None)

    elif MODEL_TYPE == 'bdt':
        # x_train, x_test, y_train, y_test =  report_and_eval.get_train_test_manual_split(x, y, train_inds, test_inds)
        model = model_selection.run_bdt(x_train, y_train, random_seed=None)

    if MODEL_TYPE in ['rf', 'bdt']:
        logging.info(f"Getting the model predictions, performance metrics, feature importances (as appropriate), and save the results to {SAVEDIR}.")
        gene_imp = report_and_eval.evaluate_interpret_save(model=model, pnet_dataset=train_dataset, model_type=MODEL_TYPE, who="train", save_dir=SAVEDIR)
        gene_imp = report_and_eval.evaluate_interpret_save(model=model, pnet_dataset=test_dataset, model_type=MODEL_TYPE, who=EVAL_SET, save_dir=SAVEDIR) # TODO: who=val/test is hardcoded
        gene_imps.append(gene_imp)

    wandb.finish()
    
logging.info(f"Saving gene importances to file in {SAVEDIR}")
# Save gene_imps to a Pickle file
with open(os.path.join(SAVEDIR, 'gene_imps.pkl'), 'wb') as file:
    pickle.dump(gene_imps, file)

if MODEL_TYPE == 'pnet':
    logging.info(f"Saving layer importances and AUCs to files in {SAVEDIR}")
    # Save layerwise_imps to a Pickle file
    with open(os.path.join(SAVEDIR, 'layerwise_imps.pkl'), 'wb') as file:
        pickle.dump(layerwise_imps, file)

    # Save aucs to a Pickle file
    with open(os.path.join(SAVEDIR, 'aucs.pkl'), 'wb') as file:
        pickle.dump(aucs, file)