# all commands to run for predict
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

NNUNet_FOLDER=$SCRIPTPATH"/nnUNet/nnunet/"
NNUNet_DATA_FOLDER="/nnunet_data"
TASK100_NAME="Task100_ATLAS_v2"
TASK103_NAME="Task103_ATLAS_v2_Self_Training"
THREADS=16

if [ `ls $NNUNet_DATA_FOLDER | wc -l` -gt 0 ]; then

    # default model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_def_fold0 -tr nnUNetTrainerV2 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_def_fold1 -tr nnUNetTrainerV2 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_def_fold2 -tr nnUNetTrainerV2 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_def_fold3 -tr nnUNetTrainerV2 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_def_fold4 -tr nnUNetTrainerV2 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 4 -z

    # dtk10 model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_dtk_fold0 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_dtk_fold1 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_dtk_fold2 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_dtk_fold3 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_dtk_fold4 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 4 -z

    # res u-net model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold0 -tr nnUNetTrainerV2_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold1 -tr nnUNetTrainerV2_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold2 -tr nnUNetTrainerV2_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold3 -tr nnUNetTrainerV2_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold4 -tr nnUNetTrainerV2_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 4 -z

    # da3 self training model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t103_da3_fold0 -tr nnUNetTrainerV2_DA3 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK103_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t103_da3_fold1 -tr nnUNetTrainerV2_DA3 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK103_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t103_da3_fold2 -tr nnUNetTrainerV2_DA3 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK103_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t103_da3_fold3 -tr nnUNetTrainerV2_DA3 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK103_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t103_da3_fold4 -tr nnUNetTrainerV2_DA3 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK103_NAME -m 3d_fullres -f 4 -z

    # ensemble high res predictions
    python $NNUNet_FOLDER"inference/ensemble_predictions.py" --npz -t $THREADS -o ensemble/predictions -f t100_def_fold0 t100_def_fold1 t100_def_fold2 t100_def_fold3 t100_def_fold4 t100_dtk_fold0 t100_dtk_fold1 t100_dtk_fold2 t100_dtk_fold3 t100_dtk_fold4 t100_res_fold0 t100_res_fold1 t100_res_fold2 t100_res_fold3 t100_res_fold4 t103_da3_fold0 t103_da3_fold1 t103_da3_fold2 t103_da3_fold3 t103_da3_fold4

fi
