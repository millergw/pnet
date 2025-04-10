# to do the germline:somatic experiment now that we've done variant-level QC filtering
python run_pnet.py --datasets somatic_amp somatic_del somatic_mut germline_mut --evaluation_set validation --model_type bdt --wandb_group bdt_somatic_and_germline_exp_004
python run_pnet.py --datasets germline_mut --evaluation_set validation --model_type bdt --wandb_group bdt_somatic_and_germline_exp_004
python run_pnet.py --datasets somatic_amp somatic_del somatic_mut --evaluation_set validation --model_type bdt --wandb_group bdt_somatic_and_germline_exp_004


# to look at stability of features in BDT across many runs, with each run's info saved to W&B
for i in {1..10};
do
  echo 'run' $i
  python run_pnet.py --datasets somatic_amp somatic_del somatic_mut --evaluation_set validation --model_type bdt --wandb_group bdt_somatic_and_germline_exp_004
done


# to look at stability of features in BDT across many runs, with each run's info saved down in easily accessible format
python run_gene_rank_stability.py


# 04/05/2024 to do the germline:somatic experiment now that we've done variant-level QC filtering
python run_pnet.py --datasets somatic_amp somatic_del somatic_mut germline_mut --evaluation_set validation --model_type pnet --wandb_group pnet_somatic_and_germline_exp_002
python run_pnet.py --datasets germline_mut --evaluation_set validation --model_type pnet --wandb_group pnet_somatic_and_germline_exp_002
python run_pnet.py --datasets somatic_amp somatic_del somatic_mut --evaluation_set validation --model_type pnet --wandb_group pnet_somatic_and_germline_exp_002



######### 
# in cancerenv environment
#########
cd cancer-net/reprod_report/
python run_pnet.torch.py --n-repeats 30 --n-hidden 6 # num_workers = 16
python gcn_variance.py # num_workers = 0
python run_pnet.torch.py --n-repeats 30 --n-hidden 6 # num_workers = 0
python run_pnet.torch.py --n-repeats 30 --n-hidden 5 # num_workers = 0



# 5/20/2024 prepping data as model input
python prep_vcfs_as_model_inputs.py --zero_impute_germline --use_only_paired --wandb_group data_prep_germline_tier12_and_somatic
