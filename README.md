# Memory Forensics to Investigate Machine Learning: A Step-by-Step Guide

#### Introduction

In this post, we’ll explore how to apply memory forensics techniques to investigate a running machine learning model. Memory forensics enables us to capture a snapshot of active memory to analyze program execution and state. For this project, we’ll focus on building and debugging a simple Convolutional Neural Network (CNN) and then capturing and analyzing it in memory using forensic tools. This post documents the step-by-step process to replicate and understand machine learning in memory forensics, complete with debugging techniques and forensic memory analysis.

---

### Part 1: Setting Up the Convolutional Neural Network (CNN)
#### Step 1: Install Necessary Packages
Before we dive into memory analysis, let’s start by setting up our environment and dependencies for building the CNN.

1. Install PyTorch and Torchvision:

```
pip3 install torch torchvision
```

2. Set Up the Project Folder:
Organize all project files under a single directory called ml_forensics_project to keep everything manageable.


#### Step 2: Building the CNN Model
To classify images using the CIFAR-10 dataset, we’ll implement a custom CNN in PyTorch, a popular framework for deep learning.

1. Define the Network Architecture: Create a Python script (e.g., cnn_cifar10.py) and define a simple CNN from scratch:

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define CNN Architecture
class SimpleCNN(nn.Module):
	def __init__(self):
    	super(SimpleCNN, self).__init__()
    	self.conv1 = nn.Conv2d(3, 6, 5)
    	self.pool = nn.MaxPool2d(2, 2)
    	self.conv2 = nn.Conv2d(6, 16, 5)
    	self.fc1 = nn.Linear(16 * 5 * 5, 120)
    	self.fc2 = nn.Linear(120, 84)
    	self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
    	x = self.pool(torch.relu(self.conv1(x)))
    	x = self.pool(torch.relu(self.conv2(x)))
    	x = x.view(-1, 16 * 5 * 5)
    	x = torch.relu(self.fc1(x))
    	x = torch.relu(self.fc2(x))
    	x = self.fc3(x)
    	return x

net = SimpleCNN()
```

2. Load and Preprocess the Data:
Load CIFAR-10 and apply necessary transformations.


```
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
```

3. Train the CNN:
Implement the training loop to train the model.
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(train_loader, 0):
    	inputs, labels = data
    	optimizer.zero_grad()
    	outputs = net(inputs)
    	loss = criterion(outputs, labels)
    	loss.backward()
    	optimizer.step()
    	running_loss += loss.item()
```
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jpgxfryztkyzudlo9prt.png)

---

### Part 2: Debugging the CNN in Memory
To capture specific states of the CNN, we’ll set up debugging breakpoints using Python Debugger (PDB) and GNU Debugger (GDB).
#### Step 1: Insert a Breakpoint in the Code
Add a line right before the forward pass to pause execution using PDB:

```
import pdb; pdb.set_trace()  # PDB breakpoint
outputs = net(inputs)
```
#### Step 2: Run the Script and Attach GDB
Open the terminal in the directory you have your script and run the Python script:

```
gdb --args python3 your_script.py
```
When the breakpoint is hit, use PDB commands to inspect objects and their fields. For example:
```
p id(variable_name)
```
This gives the memory location (in decimal). Convert it to hexadecimal using:

```
hex(id(variable_name))
```
Use Ctrl+C to get access to gdb and figure out the PID by using the command:

```
info proc
```
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jfitobhyharvhc5aw1pm.png)

Explanation of info proc Output
1. process 21416
        This is the PID (Process ID) of the running Python process.
        You can use this PID to identify and interact with the process in GDB or other tools (e.g., ps or top in Linux).

2. cmdline = '/usr/bin/python3 cnn_cifar10.py'
        This is the command line that was used to launch the process.
        It tells you that the Python script cnn_cifar10.py is being run using /usr/bin/python3.
        This is helpful if you want to ensure you're debugging the correct script or instance.

3. cwd = '/home/andrew/Desktop/ml_forensics_project'
        This is the current working directory (CWD) from where the Python script is being executed.
        It's useful if you need to check the context of file paths, ensure your script is being run from the expected directory, or inspect files being loaded by the script.

4. exe = '/usr/bin/python3.10'
        This shows the executable file that is running the Python process. In this case, it's the Python 3.10 interpreter located at /usr/bin/python3.10.
        This is helpful to know if you're running the intended version of Python and whether it's the expected environment for your script.
---

### Part 3: Capturing Memory with LiME
#### Step 1: Cloning and Setting Up LiME
LiME (Linux Memory Extractor) is a Loadable Kernel Module (LKM) that allows you to dump memory from a live Linux system. Follow these steps to clone and build LiME:
1. with the script still running and at the breakpoint, clone the LiME GitHub repository:
```
git clone https://github.com/504ensicsLabs/LiME.git
cd LiME/src
```
2. Build the LiME module:
```
make
```
3.Verify that the module has been built successfully:
```
ls -l lime*.ko
```
You should see a .ko file, which is the kernel module.

4. Load the module and specify the output file for the memory dump. Replace /path/to/dump.lime with your desired output location:
```
sudo insmod /path/to/lime-<version>.ko path=/home/username/ml_forensics_project/memdump.lime format=lime
```
Note: You need root privileges to load the module.

#### Step 2: Setting Up Volatility 3
Volatility 3 is a powerful memory forensics framework. To get started:

1. Clone the Volatility 3 repository and install its dependencies:
```
git clone https://github.com/volatilityfoundation/volatility3.git
cd volatility3
pip install -r requirements.txt
```

#### Step 3: Analyzing the Memory Image

1. Inspecting the Memory Image: Start by determining the OS profile of the memory image. This step helps Volatility understand the memory layout. Use the linux.pslist plugin to inspect running processes:
```
python3 vol.py -f /path/to/dump.lime linux.pslist.PsList
```
Note: The command above should be run inside the volatility3 directory

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/d7gpv8faz1okvmk4uzpv.png)

2. To get information about the VMAs of a particular process ID, take an ID from the running process and run this command:
```
python3 vol.py -f /path/to/dump.lime linux.proc.Map --pid <PID>
```
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ck4lp6eu8671i5kif80b.png)

---

### Part 4: Analyzing the Memory Dump with Volatility
With Volatility installed, we can inspect the memory image to locate and investigate the Python process.
#### Step 1: Find the Python Process ID in Memory
Use the pslist plugin to identify the Python process:

```
volatility -f /path/to/memdump.lime --profile=<LinuxProfile> pslist | grep python
```
[Screenshot Required: Show the output identifying the Python PID.]

#### Step 2: Analyze VMAs and Symbol Tables
1. Extract Virtual Memory Areas (VMAs) for the Python process:

```
volatility -f /path/to/memdump.lime --profile=<LinuxProfile> vma --pid=<PID>
```

2. Use the VMA output to locate where Python stores data related to its garbage collection.

### Step 3: Traversing Memory to Identify Python Objects
In this advanced step, we manually inspect the memory structure to locate objects managed by the Python garbage collector. We search for the _GC_Runtime_state structure and follow its pointers.
1. Using Volatility’s memory commands, analyze the memory areas and follow pointers:

```
# Example memory inspection command in Volatility
volatility -f /path/to/memdump.lime --profile=<LinuxProfile> memdump --pid=<PID>
```
2. Once the structure is located, write custom plugins if needed to automate object identification and capture details.
[Screenshot Required: Show memory structures of Python objects identified in the memory image.]

---
### Conclusion
In this project, we have seen how memory forensics techniques can be applied to examine the active state of a machine learning model. By training a CNN, capturing its runtime state with LiME, and analyzing it with Volatility, we gain a practical understanding of how ML processes appear in memory. This kind of forensic investigation is invaluable in fields like cybersecurity, where identifying suspicious activity in memory can reveal threats that might not be visible otherwise.

This step-by-step guide highlights both the complexity and the precision required in memory forensics. As we continue developing forensic analysis skills, investigating complex data structures within memory becomes an essential tool for understanding program behavior and system integrity.


---
### Troubleshooting tips

---
### References

1. https://sourceware.org/gdb/current/onlinedocs/gdb.pdf
2. https://volatility3.readthedocs.io/en/stable/basics.html
