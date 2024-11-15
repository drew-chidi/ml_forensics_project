# Memory Forensics to Investigate Machine Learning: A Step-by-Step Guide

#### Introduction

In this post, we’ll explore how to apply memory forensics techniques to investigate a running machine learning model. Memory forensics enables us to capture a snapshot of active memory to analyze program execution and state. For this project, we’ll focus on building and debugging a simple Convolutional Neural Network (CNN) and then capturing and analyzing it in memory using forensic tools. This post documents the step-by-step process to replicate and understand machine learning in memory forensics, complete with debugging techniques and forensic memory analysis.

---

### Part 1: Setting Up the Convolutional Neural Network (CNN)
#### Step 1: Install Necessary Packages
Before we dive into memory analysis, let’s start by setting up our environment and dependencies for building the CNN.

1. **Install PyTorch and Torchvision:**

```
pip3 install torch torchvision
```

2. **Set Up the Project Folder:**
Organize all project files under a single directory called ml_forensics_project to keep everything manageable.


#### Step 2: Building the CNN Model
To classify images using the CIFAR-10 dataset, we’ll construct a custom CNN in PyTorch, a popular framework for deep learning.

1. **Define the Network Architecture:** Create a Python script (e.g., cnn_cifar10.py) and define a simple CNN from scratch:

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


2. **Load and Preprocess the Data:**
Load CIFAR-10 and apply necessary transformations.


```
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
```

3. **Train the CNN:**
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

[Screenshot Required here]

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
Run the Python script in the terminal:

```
python3 cnn_cifar10.py
```
When PDB pauses, open a new terminal and find the PID:

```
ps aux | grep cnn_cifar10.py
```
Attach GDB to the Python process:
```
sudo gdb -p <PID>
```


[Screenshot Required: Show the PDB prompt in the first terminal and the GDB attachment in the second.]

---

### Part 3: Capturing Memory with LiME
1. **Load the LiME Kernel Module:**
Use the following command to load LiME and save memory to a file:
```
sudo insmod /path/to/lime-<version>.ko "path=/home/username/ml_forensics_project/memdump.lime format=lime"
```
[Screenshot Required: Terminal output showing the insmod command successfully creating memdump.lime.]

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
1. **Extract Virtual Memory Areas (VMAs) for the Python process:**

```
volatility -f /path/to/memdump.lime --profile=<LinuxProfile> vma --pid=<PID>
```

2. **Use the VMA output to locate where Python stores data related to its garbage collection.**

### Step 3: Traversing Memory to Identify Python Objects
In this advanced step, we manually inspect the memory structure to locate objects managed by the Python garbage collector. We search for the _GC_Runtime_state structure and follow its pointers.
1. **Using Volatility’s memory commands, analyze the memory areas and follow pointers:**

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

### References

1. https://sourceware.org/gdb/current/onlinedocs/gdb.pdf
