import subprocess as sp
import os

#this script is to run a n number of experiments and collect the data into files using the executable and the clean script

#inputs for each experiment
learningRate=float(input("Enter learning Rate:"))
cycles=int(input("Enter number of cycles to each iterations: "))
experiments=int(input("Enter the number of experiments to run: "))
savefile=input("enter file name which will be prefixed to each output: ")
os.makedirs("results", exist_ok=True)

while (experiments>0):
    #runs the cleaning.py script to scan the data and suffle the data into train and test file also normalizes it
    sp.run(["python","cleaning.py"])

    print("running experinment",experiments)
    
    prog=sp.Popen(["./main"],stdin=sp.PIPE,stdout=sp.DEVNULL,stderr=sp.DEVNULL,text=True)

    prog.stdin.write(f"{learningRate}\n")
    prog.stdin.flush()
    prog.stdin.write(f"{cycles}\n")
    prog.stdin.flush()
    prog.stdin.write(f"{savefile}{experiments}\n")
    prog.stdin.flush()

    prog.stdin.close()
    prog.wait()
    experiments -=1

print("All experiments completed! Results are saved in separate files.")