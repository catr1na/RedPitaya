# Installation
1.  Install the required python packages on your local machine

        pip install tensorflow numpy pandas (or pip3, etc, depending on versions)
    
# Create spectrograms on the RedPitaya

1.  Ssh into the RP
   
        ssh root@RED_PITAYA_IP
    
2.  Make the output directory for the spectrograms to save into

        mkdir -p /tmp/spectrograms

3.  Run spectrogram acquisition

         cd /path/to/your/code (cd NEWCODE_TEST for root@rp-f0b78c.local)
    
         /spectrogram_acquisition 1 /tmp/spectrograms

       Mode options: 0 = no save, 1 = background save, 2 = trigger save
   
       Ctrl+C to stop collecting data

 
4. Verify files were created

        ls -la /tmp/spectrograms
   
# Copy spectrograms to local machine

1. Create sync script on your local machine

        nano sync_spectrograms.sh

2. Copy the code below (with necessary changes) into sync_spectrograms.sh

        #!/bin/bash
        RED_PITAYA_IP="192.168.1.100"  #Change this!
        REMOTE_DIR="/tmp/spectrograms"
        LOCAL_DIR="$HOME/Desktop/spectrograms" 
        # Change destination as needed
        mkdir -p "$LOCAL_DIR"
        echo "Copying files from Red Pitaya..."
        scp root@$RED_PITAYA_IP:$REMOTE_DIR/*.bin "$LOCAL_DIR/"
        echo "Sync complete. Files in $LOCAL_DIR"

3. Make script executable and run

        chmod +x sync_spectrograms.sh

        ./sync_spectrograms.sh

4. Verify the files were copied

        ls -la ~/Desktop/spectrograms/

Instead of this, I assume you could use a scp command along the lines of scp root@[RED_PITAYA_IP]:/tmp/spectrograms/*.bin ~/Desktop/SPECTROGRAMS/, however, I didn’t do this so I’m not sure.

# Train the CNN

1. Get train_cnn.py from the project repository on GitHub
   
            git clone https://github.com/your_username/your_repo.git
   
    Make sure this is the proper version! Should include train_cnn.py.
   
    If not, try "git status" to check branch and "git switch "other_branch_name"" to switch          branches.

2. Place it into your working directory on your local machine

        cd your_repo (new folder that was created on your local machine by git clone)
        cp train_cnn.py ~/path/to/your/working_directory/
3. Create a project folder (if you haven’t already)
   
         python train_cnn.py (python3 train_cnn.py for my version of python)
   
      The script automatically loads your spectrogram files, creates dummy labels for testing          (replace with real labels when available), trains a CNN architecture for bubble detection,       and exports the trained weightsweights in both Keras format (*weights.h5) and C-compatible       binary file (weights_for_c.bin) for the RP


          Expected output: 
          === CNN Training Workflow ===
          Loading spectrograms from: /Users/yourname/Desktop/SPECTROGRAMS
          Found 150 spectrogram files
          Loaded 150 spectrograms with shape (150, 513, 38)
          Created 150 dummy labels

          === Model Architecture ===
          [Model summary displayed]

          Training on 120 samples, validating on 30 samples

          === Training ===
          [Training progress displayed]

          Keras weights saved to: trained_weights.h5
          Weights exported to: weights_for_c.bin


          Copy the weights back to the RP
          Copy the binary weights file
          scp weights_for_c.bin root@RED_PITAYA_IP:/tmp/
          Verify on the RP
          ls -la /tmp/weights_for_c.bin


The trained weights are now available at /tmp/weights_for_c.bin, and if on the proper branch, your C code should load these weights for inference


