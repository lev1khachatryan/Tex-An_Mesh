# Mesh alignment and rigging

We propose a new method for mesh rigging, that is based on SMPL mesh. Method is called a SMPL alignment and fitting. The method assumes that we have a not rigged mesh M1, and SMPL mesh M2. As we have mentioned in the related work, for the SMPL model we have the underlying skeleton, blend weights, T-pose mesh and other parameters. So, if we align vertices of M1 to vertices of M2 in a reasonable way, then we can take all necessary parameters to make M1animatable.

For each mesh, we find the minimum bounding box (cube) containing it by simply tracking the max/min for each X, Y and Zaxes. Using these values we find the center of the mesh as well as the extents. The width is the distance between Xmin and Xmax, the depth and height are similarly calculated depending on which axis we are using for which dimension. Then we center the mesh on the origin. Finally, we scale M1 by the size of M2 and obtain results shown in figure 1 (4).

![Alt text](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/assets/3.jpg)
**Figure 1**: The result of smpl alignment.

After alignment, for each vertex of M1 we find the nearest vertex from M2 (distance is defined by the Euclidean metric), and take its blend weights. As a joint locations for M1, we just copy all joint locations from M2. In this point we have a rigged M1 mesh, but as we use linear blend skinning for animation, then additionally we need to find the M1mesh in the rest pose. To that end, we need to notice, that M2 is obtained from SMPL rest pose mesh by applying pose-dependent transformations to each joint (remember that HMR outputs beta and theta, where each theta is the axis-angle representation of the relative rotation of joint k with respect to its parent in the kinematic tree), therefore to find the rest pose M2 mesh, we just need to apply inverse transformations to each joint of M2, and since we have aligned M1 to M2, we can apply these inverse transformations to M1 and as a result, obtain M1 in the rest pose. You can find the results in Figure 2.

![Alt text](https://github.com/lev1khachatryan/Tex-An_Mesh/blob/master/assets/4.jpg)
**Figure 2**: The result of smpl alignment.


The necessary pretrained models can be found [here](https://drive.google.com/open?id=10R_hXb7YyJgWpRkYA8FBsHCEsLLjq2yb)
