# Q-Net
A pytorch implementation of the paper : ["Learning with learned loss function: Speech enhancement with quality-net to improve perceptual evaluation of speech quality"](https://ieeexplore.ieee.org/abstract/document/8902088)

## 0. Brief Summary of the paper
* Speech enhancement model should be designed to maximize PESQ, not to reduce MSE criterion. 
* Since PESQ posseses non-differentiable properties, PESQ may not be used directly to optimize speech enhancement model.
* Instead, train 'surrogate model' which is differentiable and learned from training data to eventually approximate PESQ function. The name of the surrogate model is 'Q-Net' (Quality-Net) 
* After training Q-net, concate the model to the original speech enhancement model, fix the Q-Net parameter, and fine-tune the speech enhancement model.   
 ![Q-Net](https://user-images.githubusercontent.com/77431192/117633167-c0b33180-b1b8-11eb-99b3-c012f036bf2a.PNG)  
 *Note: In this repository, We only provide source code of step1 (Training Q-Net, `train.py`). We conducted step2 experiment using our own speech enhancement model, which is not released to public yet.* 
 
 ## 1. Prepare Dataset
 Following is the procedure of preparing dataset proposed in the paper. 
 * All dataset is derived from TIMIT dataset. 
 * Each data should be in a form ([degraded speech, clean speech], PESQ Score)
 * Train : randomly select 300 clean wav files from TIMIT training -> corrupt with 10 noise types, 5 SNR levels. 
 * Valid : randomly select 100 clean wav files from TIMIT training -> corrupt with unseen noise, 5 SNR levels.
 * Test: randomly select 100 clean wav files from TIMIT test -> corrupt with 4 uneseen noises, 5 SNR levels.
 
  *Note: We added 10000 more data pair of which degraded speeech is derived from our own SE model to let Q-net model to learn a tendency of the SE model.*  
  
**Histogram of final dataset (X-axis = PESQ, Y-axis = number of data)**   
  ![dataset](https://user-images.githubusercontent.com/77431192/117650336-8acb7880-b1cb-11eb-8fab-0d32fb51c3bb.PNG)


 ## 2. Q-Net Training Results
 ~~~
 python qnet_infer.py
 ~~~
 Results of plotting all points (true PESQ, inferred PESQ) using test dataset  
 ![training_result](https://user-images.githubusercontent.com/77431192/117636134-7b443380-b1bb-11eb-972c-2f0292cffcc2.PNG)

 ## 3. SE Model Fine-tuning Results
 We fine-tuned existing SE model after concatenating Q-net, but results was poor. PESQ slightly improved, while other metric such as SDR degraded. Main reason I guess is the data tendency discrepancy. While Q-Net model is trained with TIMIT dataset as proposed in the thesis, We fine-tuned SE model with LRS dataset as we usually did before. 
 
 We also tried other fine-tuning method such as GAN-style method first proposed in [Fu, Szu-Wei, et al. "Boosting Objective Scores of a Speech Enhancement Model by MetricGAN Post-processing." 2020 ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9306484&tag=1), and used stabilization techniques proposed in [Kawanaka, Masaki, et al. "Stable training of DNN for speech enhancement based on perceptually-motivated black-box cost function." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054578) However, observing loss per epoch graph,  the model dosen't seem converge. Moreover, observing how parameters of Q-Net modified with GAN-style training (SE model - Q-Net relationship equals to Generator - Discrimiator relationship), Q-Net seems to infer only 0.6-0.7 as a scaled PESQ because PESQ of most of data pair [clean, enhanced] from SE model is around 3 (0.6-0.7 if scaled). Refer to the graph below.   
 ![GAN_train_result](https://user-images.githubusercontent.com/77431192/117656757-af2b5300-b1d3-11eb-8c32-8e112c74b86a.PNG)

