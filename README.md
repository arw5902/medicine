This repository presents a live medication monitoring and advisory system that detects medicine intake behavior in real time using video input. The system leverages a ceiling-mounted indoor camera and applies the TimeSformer (GitHub https://github.com/facebookresearch/TimeSformer) video-based attention architecture to identify moments of medicine-taking from continuous video streams.
This repository has two parts: the first part is the TimeSformer code; the second part consists of the scripts for medication behavior detection. Note default values are provided for all scripts, check the comments at the beginning of each file to customize them.
Note: all videos and clips are not uploaded to GitHub due to privacy purpose.
Follow the steps below to generate video files and run TimeSformer to obtain the model file. The commands should be executed from the MedicationDetector directory unless otherwise stated.
1. Record medication-taking training video samples one at a time:  
	python RecordVideo.py
For example, the videos are named to begin with “a” in the MedicationDetector/videos directory. Note that these videos are not uploaded to GitHub for privacy purposes.
2. Divide each video into clips for each of the clip size parameters (75/100/125):  
	python VideoClips.py --clipSize {75|100|125}
The batch command for this is written in callPScripts.sh; just specify the minimum and maximum video_ids in the script.
3. For each video, find the frame number of taking medicine by examining the clips. Create the ground truth .csv file called vsamples.csv, and record the medicine-taking frame number in it. Place vsamples.csv in the MedicationDetector/csv/ folder.
4. Label each clip according to the ground truth .csv file. This should also be run for each of the clip size parameters (75/100/125):  
	python ClipLabel.py --clipSize {75|100|125}
A csamples.csv file will be created in the MedicationDetector/{clip_size}/ folder.
5. Make sure the csamples.csv file generated above resides in the machine that will run TimeSformer (it may need to be copied over if another machine is used for video recording and clipping). It should be placed in the {base}/{clip_size}/ folder. The example {base} is
“/home/home/Documents/MedicineDetector/output/”. Run the following command from the TimeSformer directory to split csamples.csv into train, val, and test as required by TimeSformer. Run this for each of the clip size parameters (75/100/125):  
	python ds_split.py --clipSize {75|100|125}
Three files: train.csv, val.csv, and test.csv, will be created in the {base}/{clip_size}/ folder.
6. Make sure the video clips are copied over to the {base}/{clip_size}/ folder on the machine that will run TimeSformer. The example {base} is
“/home/home/Documents/MedicineDetector/output/”.
7. From the machine that contains TimeSformer code, run the following script from TimeSformer directory for each of the clip size parameters (75/100/125):  
	cmd_ts_{clip_size}.sh
This will start TimeSformer according to the parameters set in the above script. Check TimeSformer’s GitHub for further information about each parameter. NUM_GPUS is set to 2 to represent the number of GPUs used to run TimeSformer.
8. Check TimeSformer’s log file stdout.log for model accuracy for each of the clip size parameters (75/100/125). Save the best model file in the TimeSformer/checkpoints directory for each run.
Follow the steps below to test the model file. These steps do not have to be executed on a machine with a high-end GPU, but the TimeSformer code is required for scripts to make predictions using the model. The commands should be executed from the MedicationDetector directory unless otherwise stated.
1. Copy over the best model file for each clip size (75, 100, and 125) to the corresponding directory inside the project, for example: MedicationDetector/model/75/checkpoint_epoch_00014.pyth.
2. Run the following command for all clip sizes (75, 100, and 125) to generate the .csv file for the ground truth of test.csv (15% of all clips sent to TimeSformer) and the model prediction side by side:  
	python inf_test.py --clipSize {75|100|125}
This script will generate test_75_ret.csv, test_100_ret.csv, and test_125.csv after running for each clip size.
3. Draw a plot based on ground truth vs. model predictions on the test dataset for all three clip window sizes; that is, the .csv files obtained from step 2: test_75_ret.csv, test_100_ret.csv, and test_125_ret.csv. This execution also plots the precision-recall curve (PR curve), including the values of precision, recall, threshold, F1_score, and PR AUC for all three clip window sizes.  
	python drawGraphTestCsv.py
4. Record the holdout test videos in the videos folder. For example, the videos are renamed to begin with “t” in the MedicationDetector/videos directory.  
	python RecordVideo.py --outputFname t1_.mp4
5. Divide the holdout test video into clips for each of the clip size parameters (75/100/125). Note that these videos are also not uploaded to GitHub for privacy purposes.  
	python VideoClips.py --filePath t1_.mp4 --clipSize {75|100|125}
6. Check if the holdout test video contains the medication-taking behavior (the environment should be set up properly as instructed in the TimeSformer GitHub). Run the command below for each of the clip size parameters (75/100/125):  
	python inf.py --videoName t1 --clipSize {75|100|125}
A .csv file (e.g. csv/holdout/frames_75/csamples_live_t1.csv) will be generated to record the likelihood of medicine-taking for each clip of the holdout test video. A detected medication behavior will be recorded in csv/MedicationRecord.csv.
The batch command for this is written in callPScripts_live.sh; just specify the minimum and maximum holdout video_ids in the script, uncomment the corresponding command, and comment out other commands.
7. Label each clip of a holdout test video according to a ground truth .csv file, such as csv/holdout/vsamples_fullvideo.csv. Run the command below for each of the clip size parameters (75/100/125):  
	python ClipLabel_live.py --videoName t1 --clipSize {75|100|125}
The batch command for this is written in callPScripts_live.sh; just specify the minimum and maximum holdout video_ids in the script.
8. Draw a batch plot based on csamples_live_t##.csv to compare the performance of three clip window sizes: 75, 100, and 125.  
	python drawGraph.py
9. Obtain the model accuracy for full video detection with all three clip sizes:  
	python drawGraphThreshold.py
