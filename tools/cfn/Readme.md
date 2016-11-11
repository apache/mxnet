# **Distributed Deep Learning Made Easy**
**Authors**: Naveen Swamy, Joseph Spisak

Machine learning is a field of computer science that enables computers to learn without being explicitly programmed. It focuses on algorithms that can learn from and make predictions on data. 

Most recently, one branch of machine learning, called deep learning, has been deployed successfully in production with higher accuracy than traditional techniques, enabling capabilities such as speech recognition, image recognition, and video analytics. This higher accuracy comes, however, at the cost of significantly higher compute requirements for training these deep models. 

One of the major reasons for this rebirth and rapid progress is the availability and democratization of cloud-scale computing. Training state-of-the-art deep neural networks can be time-consuming, with larger networks like [ResidualNet](https://arxiv.org/abs/1512.03385) taking several days to weeks to train, even on the latest GPU hardware. Because of this, a scale-out approach is required.  

Accelerating training time has multiple benefits, including:  

* Enabling faster iterative research, allowing scientists to push the state of the art faster in domains such as computer vision or speech recognition. 
* Reducing the time-to-market for intelligent applications, allowing AI applications that consume trained, deep learning models to access newer models faster.
* Absorbing new data faster, helping to keep deep learning models current.

[AWS CloudFormation](https://aws.amazon.com/cloudformation), which creates and configures Amazon Web Services resources with a template, simplifies the process of setting up a distributed deep learning cluster. The CloudFormation Deep Learning template uses the [Amazon Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB) (supporting MXNet, TensorFlow, Caffe, Theano, Torch, and CNTK frameworks) to launch a cluster of [Amazon EC2](https://aws.amazon.com/ec2) instances and other AWS resources needed to perform distributed deep learning. CloudFormation creates all resources in the customer account.  

# EC2 Cluster Architecture 
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide1.png)

# Resources created by the Deep Learning template
The Deep Learning template creates a stack that contains the following resources:  

* A VPC in the customer account. 
* The requested number of worker instances in an [Auto Scaling](https://aws.amazon.com/autoscaling) group within the VPC. These worker instances are launched in a private subnet. 
* A master instance in a separate Auto Scaling group that acts as a proxy to enable connectivity to the cluster via SSH. CloudFormation places this instance within the VPC and connects it to both the public and private subnets. This instance has both public IP addresses and DNS. 
* A security group that allows external SSH access to the master instance. 
* Two security groups that open ports on the private subnet for communication between the master and workers. 
* An [IAM](https://aws.amazon.com/iam) role that allows users to access and query Auto Scaling groups and the private IP addresses of the EC2 instances. 
* A NAT gateway used by the instances within the VPC to talk to the outside world. 

The startup script enables SSH forwarding on all hosts. Enabling SSH is essential because frameworks such as MXNet makes use of SSH for communication between master and worker instances during distributed training. The startup script queries the private IP addresses of all the hosts in the stack, appends the IP address and worker alias to /etc/hosts, and writes the list of worker aliases to /opt/deeplearning/workers.  

The startup script sets up the following environment variables: 

* **$DEEPLEARNING_WORKERS_PATH**: The file path that contains the list of workers  

* **$DEEPLEARNING_WORKERS_COUNT**: The total number of workers  

* **$DEEPLEARNING_WORKER_GPU_COUNT**: The number of GPUs on the instance  

# Launch a CloudFormation Stack
**Note:**  To scale to the desired number of instances beyond the [default limit](https://aws.amazon.com/ec2/faqs/#How_many_instances_can_I_run_in_Amazon_EC2), file a [support request](https://aws.amazon.com/contact-us/ec2-request).

1. Download the Deep Learning template from the [MXNet](https://github.com/dmlc/mxnet/tree/master/tools/cfn) GitHub repo.

2. Open the [CloudFormation console](https://console.aws.amazon.com/cloudformation), and then choose **Create New Stack**. 
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide2.png)  

3. Choose **Choose File** to upload the template, and then choose **Next**:
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide3.png)  

4. For **Stack name**, enter a descriptive stack name.

5. Choose a GPU **InstanceType**, such as a [P2.16xlarge](https://aws.amazon.com/ec2/instance-types/p2/).  

6. For **KeyName**, choose an EC2 key pair.  

7. For **SSHLocation**, choose a valid CIDR IP address range to allow SSH access to the master instance and stack.  

8. For **Worker Count**, type a value. The stack provisions the worker count + 1, with the additional instance acting as the master. The master also participates in the training/evaluation. Choose **Next**.
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide4.png)

9. (Optional) Under **Tags**, type values for **Key** and **Value**. This allows you to assign metadata to your resources.
   (Optional) Under **Permissions**, you can choose the IAM role that CloudFormation uses to create the stack. Choose **Next**.
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide5.png)

10. Under **Capabilities**, select the checkbox to agree to allow CloudFormation to create an IAM role. An IAM role is required for correctly setting up a stack.
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide6.png)  

11. To create the CloudFormation stack, choose **Create**
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide7.png)  

12. To see the status of your stack, choose **Events**. If stack creation fails, for example, because of an access issue or an unsupported number of workers, troubleshoot the issue. For information about troubleshooting the creation of stacks, see [Troubleshooting AWS CloudFormation](http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/troubleshooting.html). The event log records the reason for failure. 

# Log in to the master instance.
SSH agent forwarding securely connects the instances within the VPC that is connected to the private subnet. The idea is based on [Securely Connect to Linux Instances Running in a Private Amazon VPC.](https://aws.amazon.com/blogs/security/securely-connect-to-linux-instances-running-in-a-private-amazon-vpc/)

1. **Find the public DNS/IP of the master.**  

The CloudFormation stack **output** contains the Auto Scaling group in which the master instance is launched. Note the Auto Scaling group ID for MasterAutoScalingGroup.  
![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tools/cfn/Slide8.png)

  a. Open the [Amazon EC2 console](https://console.aws.amazon.com/ec2).  
    
  b. In the navigation pane, under **Auto Scaling**, choose **Auto Scaling Groups**.  

  c. On the **Auto Scaling** page, search for the group ID and select it.  

  d. On the **Instances** tab, find the instance ID of the master instance.  

  e. Choose the instance to find the public DNS/IP address used for login.  

2. **Enable SSH agent forwarding.**

This enables communication with all instances in the private subnet. Using the DNS/IP address from Step 1, modify the SSH configuration to include these lines: 

    Host IP/DNS-from-above  
    ForwardAgent yes

3. **Run MXNet distributed training.**  

The following example shows how to run MNIST with data parallelism. Note the use of the DEEPLEARNING_* environment variables:  

	#terminate all running Python processes across workers 
	while read -u 10 host; do ssh $host "pkill -f python" ; done 10<$DEEPLEARNING_WORKERS_PATH  
	
	#navigate to the mnist image-classification example directory  
	cd ~/src/mxnet/example/image-classification  
	
	#run the MNIST distributed training example  
	../../tools/launch.py -n $DEEPLEARNING_WORKERS_COUNT -H $DEEPLEARNING_WORKERS_PATH python train_mnist.py --gpus $(seq -s , 0 1 $(($DEEPLEARNING_WORKER_GPU_COUNT - 1))) --network lenet --kv-store dist_sync

These steps are only a subset. For more information about running distributed training, see [Run MXNet on Multiple Devices](http://mxnet.readthedocs.io/en/latest/how_to/multi_devices.html). 

#FAQ

###1. How do I change the IP addresses that are allowed to SSH to the master instance?
The CloudFormation stack output contains the security group that controls the inbound IP addresses for SSH access to the master instance. Use this security group to change your inbound IP addresses.  

###2. When an instance is replaced, are the IP addresses of the instances updated? 
No. You must update IP addresses manually.  

###3. Does the master instance participate in training/validation?
Yes. Because most deep learning tasks involve GPUs, the master instance acts both as a proxy and as a distributed training/validation instance.

###4. Why are the instances in an Auto Scaling group? 
[Auto Scaling](https://aws.amazon.com/autoscaling/) group maintains the number of desired instances by launching a new instance if an existing instance fails. There are two Auto Scaling groups: one for the master and one for the workers in the private subnet. Because only the master instance has a public endpoint to access the hosts in the stack, if the master instance becomes unavailable, you can terminate it and the associated Auto Scaling group automatically launches a new master instance with a new public endpoint. 

###5. When a new worker instance is added or an existing instance replaced, does CloudFormation update the IP addresses on the master instance?
No, this template does not have the capability to automatically update the IP address of the replacement instance.
