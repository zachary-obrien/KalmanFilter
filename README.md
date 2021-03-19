# KalmanFilter

## Installation Instructions
This application processes video and estimates the velocity of the object detected by the Nerual Network. While the current demo is of a ball moving, the model is relaceable with whatever detection model you'd like.

### Apt-get Installs
```
sudo apt-get install git
sudo apt-get install vim
sudo apt-get install python3-pip
sudo apt-get upgrade
sudo apt-get install protobuf-compiler
```

### Pip Installs
```
sudo pip3 install ––upgrade pip
pip install testresources
pip install keras

#tensorflow-gpu is automatically included with tensorflow2
pip install tensorflow

EITHER:<br/><br/>
  pip3 install setuptools
  pip3 install --upgrade setuptools
 ```
### Other Installs
```
git clone https://github.com/tensorflow/models.git
cd models/research
#Compile protos
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

python object_detection/builders/model_builder_tf2_test.py
```

See License File for use Permissions
