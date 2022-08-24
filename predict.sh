#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

export nnUNet_raw_data_base=$SCRIPTPATH"/nnUNet_raw/"
export nnUNet_preprocessed=$SCRIPTPATH"/nnUNet_preprocessed/"
export RESULTS_FOLDER=$SCRIPTPATH"/nnUNet_trained_models/"

python $SCRIPTPATH/copy_images_to_nnunet_format.py

mkdir -p t100_def_fold0 t100_def_fold1 t100_def_fold2 t100_def_fold3 t100_def_fold4
mkdir -p t100_dtk_fold0 t100_dtk_fold1 t100_dtk_fold2 t100_dtk_fold3 t100_dtk_fold4
mkdir -p t100_res_fold0 t100_res_fold1 t100_res_fold2 t100_res_fold3 t100_res_fold4
mkdir -p t103_da3_fold0 t103_da3_fold1 t103_da3_fold2 t103_da3_fold3 t103_da3_fold4
mkdir -p ensemble/predictions

bash $SCRIPTPATH/predict_all_folds.sh

python $SCRIPTPATH/rename_predictions.py

rm -rf t100_def_fold0 t100_def_fold1 t100_def_fold2 t100_def_fold3 t100_def_fold4
rm -rf t100_dtk_fold0 t100_dtk_fold1 t100_dtk_fold2 t100_dtk_fold3 t100_dtk_fold4
rm -rf t100_res_fold0 t100_res_fold1 t100_res_fold2 t100_res_fold3 t100_res_fold4
rm -rf t103_da3_fold0 t103_da3_fold1 t103_da3_fold2 t103_da3_fold3 t103_da3_fold4

