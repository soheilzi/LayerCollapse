# download and extract imagenet datasset into /data/soheil/datasets/imagenet

# Download and extract imagenet dataset
mkdir -p /data/soheil/datasets/imagenet
cd /data/soheil/datasets/imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
tar -xvf ILSVRC2012_img_train.tar
tar -xvf ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_test.tar
tar -xvf ILSVRC2012_devkit_t12.tar.gz
rm ILSVRC2012_img_train.tar
rm ILSVRC2012_img_val.tar
rm ILSVRC2012_img_test.tar
rm ILSVRC2012_devkit_t12.tar.gz

