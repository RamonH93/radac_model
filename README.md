# Instructions

## Initialize identity server with Docker
```bash
$ git clone https://github.com/RamonH93/radac_model.git 
$ cd radac_model 
$ docker load --input wso2is.tar 
$ docker volume create wso2is_data 
$ docker run -d -p 9443:9443 --name wso2is -v wso2is_data:/home/wso2carbon wso2is:latest 
```
Wait for startup to complete.. <br> 
Visit https://localhost:9443/carbon to verify startup is successful and complete. <br>
<br>
Enable the admin service to allow SOAP communication with the PAP and PDP:
```bash
$ docker exec wso2is /bin/sh -c "echo '\n[admin_service.wsdl]\nenable = true' >> wso2is-5.10.0/repository/conf/deployment.toml"
$ docker restart wso2is
```

Wait for startup to complete.. <br> 
Visit https://localhost:9443/carbon and login with admin/admin to verify the identity server is running correctly.

<hr >

## Prepare Python environment

TensorFlow 2.1.0 requires <b>python>=3.5,<=3.7.*</b>, make sure your python version fits the requirements! (I use 3.7.7)

```bash
$ python --version
Python 3.7.7
$ python -m pip install -r requirements.txt
```

<hr>

## *[Optional]* Install TensorFlow-GPU:

See list of [CUDA®-enabled GPU cards](https://developer.nvidia.com/cuda-gpus)

The following NVIDIA® software must be installed on your system:

>* [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) (CUDA® 10.1 requires 418.x or higher.)
>* [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (TensorFlow supports CUDA® 10.1 (TensorFlow >= 2.1.0))
>* [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA® Toolkit.
>* [cuDNN](https://developer.nvidia.com/cudnn) SDK 7.6
>* [optional] [TensorRT 6.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) to improve latency and throughput for inference on some models (I did not use this)

For additional help with setup, setting up paths, etc, please refer to https://www.tensorflow.org/install/gpu.
```bash
$ python -m pip install tensorflow-gpu==2.1.0
```
*[Optional]* For CUPTI to work best on Windows, it is advisable to allow access to GPU performance counters to all users:

<pre>
1. Launch the NVIDIA Control Panel as an administrator.
2. Either
    * Log in to Windows as an administrator, then launch the NVIDIA Control Panel, or  
    * Use the Windows file manager to navigate to C:\Program Files\NVIDIA Corporation\Control Panel Client, then right-click nvcplui.exe and select Run as administrator.
3. If the Developer module is not visible, then click Desktop from the menu bar and check Enable Developer settings.
4. From the NVIDIA Control Panel, navigate to Developer->Manage GPU Performance Counters.
5. Select Allow access to GPU performance counters to all users, then click Apply.
</pre>
