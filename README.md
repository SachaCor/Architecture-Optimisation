Here is the project to make a NN work using BREVITAS:

Files:
In "bnn_pynq_train.py" are all the options you can add when calling the line with the terminal
In "trainer.py" is the code with all the functions and classes used to make the NN train and test
In "Automation_smth.sh" are bash files to automize works for long repetitive studies

Folders:
"cfg" gets the .ini file with the architecture you want to use
"exp" has the folders where are gathered all the informations for each training and testing sessions. They are sorted by date and time of starting computation
"images" has the different datasets
"models" has python functions for the use of the code
"__pycache__" has other functions

Files in each "exp" folder:
"log.txt" is simply the log
"loss_test.txt" is the loss value for the last testing session done
"loss_training.txt" gathers all the loss values during the training
"matrix_noisy.txt" gets the matrix of all the guesses made by the network during the testing session
"table_noisy" summarizes the output guess vector and final guess for each of the tested image
"test.txt" summarizes the functions used, the architecture, and the values for each weight and bias from the trained network


to run a training session you have to write in the terminal the right folder:
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 1000 --experiments exp --gpus None
you can add options, regarding the ones summed in the "bnn_pynq_train.py" file

For the testing: 
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /path/to/folder/bnn_pynq_/exp/yourfolder/checkpoints/best.tar


