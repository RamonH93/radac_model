# Instructions
## Generate dataset
### Initialize PDP Server with Docker
 Build from Dockerfile because it is modded to correct timezone <br> <br>
<code>
\> cd docker-is/dockerfiles/ubuntu/is <br>
\> docker build -t wso2is:5.10.0 . <br>
\> docker volume create wso2is_data  <br>
\> docker run -d -p 9443:9443 --name wso2is -v wso2is_data:/home/wso2carbon/wso2is-5.10.0 wso2is:5.10.0 <br> <br>
</code>
Wait ~70 secs for startup to complete <br> <br>
<code>
\> docker exec wso2is /bin/sh -c "echo '\n[admin_service.wsdl]\nenable = true' >> wso2is-5.10.0/repository/conf/deployment.toml" <br>
\> docker restart wso2is
</code>
<br> <br>
Wait ~70 secs for startup to complete

Visit <url>https://localhost:9443/carbon</url> and loging with admin/admin to verify the PDP Server is running correctly.

## Prepare tensorflow-gpu==2.1.0
!! MUST HAVE NVIDIA GPU

The following NVIDIA® software must be installed on your system:

>* NVIDIA® GPU drivers —CUDA® 10.1 requires 418.x or higher.
>* CUDA® Toolkit —TensorFlow supports CUDA® 10.1 (TensorFlow >= 2.1.0)
>* CUPTI ships with the CUDA® Toolkit.
>* cuDNN SDK 7.6

<br>
<code>> python3 -m pip install -r requirements.txt</code> <br>
<br>
For CUPTI to work properly, it is advisable to allow access to GPU performance counters to all users:

1. Launch the NVIDIA Control Panel as an administrator.

2. Either

    * Log in to Windows as an administrator, then launch the NVIDIA Control Panel, or  

    * Use the Windows file manager to navigate to C:\Program Files\NVIDIA Corporation\Control Panel Client, then right-click nvcplui.exe and select Run as administrator.

3. If the Developer module is not visible, then click Desktop from the menu bar and check Enable Developer settings.

4. From the NVIDIA Control Panel, navigate to Developer->Manage GPU Performance Counters.

5. Select Allow access to GPU performance counters to all users, then click Apply.
