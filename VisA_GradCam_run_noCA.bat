call C:/Users/MVCLAB/Anaconda3/Scripts/activate.bat C:/Users/MVCLAB/Anaconda3/envs/gradCam
cd C:/Users/MVCLAB/Desktop/GradCam/pytorch-grad-cam
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/capsules" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_capsules_noCA/noCA_capsules_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/candle" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_candle_noCA/noCA_candle_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/cashew" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_cashew_noCA/noCA_cashew_model.pkl"
@REM python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/chewinggum" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_chewinggum_noCA/noCA_chewinggum_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/fryum" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_fryum_noCA/noCA_fryum_model.pkl"
@REM python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/macaroni1" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_macaroni1_noCA/noCA_macaroni1_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/macaroni2" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_macaroni2_noCA/noCA_macaroni2_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/pcb1" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_pcb1_noCA/noCA_pcb1_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/pcb2" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_pcb2_noCA/noCA_pcb2_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/pcb3" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_pcb3_noCA/noCA_pcb3_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/pcb4" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_pcb4_noCA/noCA_pcb4_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/TEST/pipe_fryum" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/noCA/experiment_pipe_fryum_noCA/noCA_pipe_fryum_model.pkl"
@pause

