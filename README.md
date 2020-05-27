# Tex-An Mesh: Textured and animatable human body mesh reconstruction from a single image

We provide a method that will surely excite every Potterhead on the planet. As we all remember, the people in the photos hanging on the fictional wall of Hogwarts freely move around and even jump into other frames to visit. Our method helps to make the magic a reality, so we can make a person in a 2-D photo perform various motions.


![Alt text](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/assets/1.jpg)
**Figure 1**: Given a single photo as input (far left), we create a 3D animated version of the subject, which can now walk towards the viewer (middle).

In this project, we address the problem of reconstructing a fully textured and animatable human body mesh from only a single image. It has many applications ranging from virtual and augmented reality to the production of movies and video games. 

Our contributions are the following:

* we obtain a fully textured and animatable mesh by combining textured mesh and SMPL predictions,

* to the best of our knowledge, this is the first work which makes the predicted fully textured human body mesh animatable.


## Method

Given a single image, our goal is to reconstruct the underlying 3D fully textured and rigged mesh of the human body. The overall system works as shown in figure 2. At first, we apply state-of-the-art algorithms to perform person segmentation and image inpainting. Then we use human mesh recovery (HMR) method to obtain shape and pose parameters for the SMPL model. For textured mesh reconstruction we use pixel aligned implicit function (PIFu). Finally, obtained two meshes are aligned to get skeleton and blend weight for textured mesh.

![Alt text](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/assets/2.jpg)
**Figure 2**: Overview of our method.

***Unlike existing methods, which give partially textured meshes or synthesize the back regions based on frontal views, our approach, owing to PIFu, gives fully textured mesh.***

## Results

Peter Dinklage             |  Vladimir Putin
:-------------------------:|:-------------------------:
![](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/assets/dinklage.gif)  |  ![](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/assets/putin.gif)


You can find the documentation of the project [here](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/DOCUMENTATION.pdf)

<br>
<br>
<br>
<br>

## My Running Environment
<b>Hardware</b>
- CPU: Intel® Core™ i9-9900K CPU @ 3.60GHz × 16 
- GPU: NVIDIA® GeForce RTX 2080 Ti/PCIe/SSE2
- Memory: 62GB GiB
- OS type: 64-bit
- Disk: 1.2 TB

<b>Operating System</b>
- ubuntu 18.04 LTS

<b>Software</b>
- Python 3.6.2
- TensorFlow 1.4.0
- CUDA 10.0
- cuDNN 7.0

## Contact
If there's some suggestions you would like to give or if you're just feeling social,
feel free to [email](mailto:levon.khachatryan.1996.db@gmail.com) me or connect with me on [LinkedIn](https://www.linkedin.com/in/levonkhachatryan/).