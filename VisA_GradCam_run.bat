call C:/Users/MVCLAB/Anaconda3/Scripts/activate.bat C:/Users/MVCLAB/Anaconda3/envs/gradCam
cd C:/Users/MVCLAB/Desktop/GradCam/pytorch-grad-cam
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/capsules" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_capsules/capsules_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/candle" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_candle/candle_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/cashew" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_cashew/cashew_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/chewinggum" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_chewinggum/chewinggum_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/fryum" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_fryum/fryum_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/macaroni1" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_macaroni1/macaroni1_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/macaroni2" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_macaroni2/macaroni2_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/pcb1" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_pcb1/pcb1_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/pcb2" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_pcb2/pcb2_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/pcb3" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_pcb3/pcb3_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/pcb4" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_pcb4/pcb4_model.pkl"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/tools/Resize/VisADataSet_Resize_test_good/pipe_fryum" --method "gradcam" --source_file_path "C:/Users/MVCLAB/Desktop/DRA/experiment/HolisticDilatedwithCA/experiment_pipe_fryum/pipe_fryum_model.pkl"
@pause

