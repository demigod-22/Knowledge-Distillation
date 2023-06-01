# KD
Implementation of knowledge distillation

With Multiple teachers : 
  1) Each teacher model trained by a different sensor.
       a) acct.py
       b) gyrt.py
       c) magt.py
       the above three files are present in the Student Models
       
There are two modules in this architecture.

The first one is in DAE and the second in Student Models(the above teacher models)


Steps to be followed : 

1)Run the  file present in DAE

2)Run the acct.py   gyrt.py   magt.py files present in student models

3)After running/training the teacher models , run the student models a)accs.py b)gyrs.py c)mags.py files also present in Student Models

4)Now the three student models have been trained

5)Run the Ensemble.py file in Student Models to get the combined output when the PAMAP2 dataset is given.

