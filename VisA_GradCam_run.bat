call C:\Users\MVCLAB\Anaconda3\Scripts\activate.bat C:\Users\MVCLAB\Anaconda3\envs\gradCam
cd C:\Users\MVCLAB\Desktop\GradCam\pytorch-grad-cam

python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/candle/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/capsules/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/cashew/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/chewinggum/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/fryum/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/macaroni1/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/macaroni2/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/pcb1/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/pcb2/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/pcb3/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/pcb4/test/anomaly" --method "gradcam"
python cam.py --image_file_path "C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/pipe_fryum/test/anomaly" --method "gradcam"
@pause 