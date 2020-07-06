# PR2 teleoperation using PerceptionNeuron

This repository subscribes rostopic from virtual machine, publish tfs of human motions, and teleop PR2 robots.

## Network
### For machines with 134.100.13.XXX IPs
- Install a win7 virtual machine and setup your Win7 and Unbuntu environment following [Running instructions] (https://gogs.crossmodal-learning.org/shuang.li/perception_neuron_ros-master).
- Setup network environment of your machine following [Via the network gateway (preferred for machines with 134.100.13.XXX IPs)] (https://gogs.crossmodal-learning.org/TAMS/tams_pr2/wiki/Network/) in [tams_pr2](https://gogs.crossmodal-learning.org/TAMS/tams_pr2) wiki page.

  - Test if your machine to Trixi's computers.

  ```
  export ROS_MASTER_URI=http://10.68.0.1:11311
  rostopic echo /joint_states
  ```

### For basestation and c1
- Test if imu data publishes to Trixi's computers.
```
rostopic echo /perception_neuron/data_1
```

### WIFI connection
- Plug a wifi adapter to your computer, and connect to [INFFUL] or [PerceptionNeuron TAMS] in the virtual machine.
- Set up wifi connection of PerceptionNeuron followed page 28 in [perception neuron manual] (https://neuronmocap.com/system/files/software/Axis%20Neuron%20User%20Manual_V3.8.1.5.pdf).
- In case the wifi connection of PerceptionNeuron always fails, you can disable the local network of your virtual machine first then connect to wifi, then enable the local network again. We need the local network to transfer data from the virtual machine to Linux platform.
