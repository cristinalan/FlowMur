# Source code of FlowMur
Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan and Elisa Bertino, "[FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a148/1Ub245RZpo4)," IEEE S&P, 2024.
## The workflow of FlowMur
<div align="center">
    <img src="./Workflow.png" width="40%">
</div>

## How to start
This example is for the following setting:

dataset --> Google Speech Command Dataset V2

target model --> SmallCNN；surrogate model --> LargeCNN

#class of $D$ --> 10; #class of $D_{aux}$ --> 25; #class of $D_{sur}$ --> 26

### Step 1: Data Preprocessing
Extract audio features for $D$ and $D_{sur}$ respectively.
```shell
python data_preprocessing.py
```
### Step 2: Obtain the Surrogate Model
Train the surrogate model on $D_{sur}$
```shell
python Benign_Model.py
```
### Step 3: Generate the Trigger
Optimize the trigger on the surrogate model
```shell
python generate_trigger.py
```
### Step 4: Data Poisoning and Backdoor Injection
Poison $D$ and train SmallCNN on poisonous $D$
```shell
python Attack.py
```
## Comments
If you have any questions about the code, please feel free to ask here or contact me via email at <jhlan16@stu.xidian.edu.cn>.
